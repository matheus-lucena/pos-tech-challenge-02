"""
VRP Genetic Algorithm operators: selection, crossover, mutation, and local search.
This module contains all genetic operators and local optimization methods.
"""

import random
import math
from copy import deepcopy
from typing import List, Tuple, Callable

from vrp.config import (DEPOT_INDEX, TOURNAMENT_SIZE, TWO_OPT_MUTATION_PROBABILITY,
                    MAX_ITER_WITHOUT_IMPROVE_INTER_ROUTE, MAX_IMPROVEMENTS_2OPT)

class VRPOperators:
    """Container class for all VRP genetic algorithm operators."""
    
    def __init__(self, max_vehicles: int, vehicle_max_points: int, 
                 fitness_function: Callable[[List[List[int]]], float]):
        """
        Initialize VRP operators.
        
        Args:
            max_vehicles: Maximum number of vehicles
            vehicle_max_points: Maximum points per vehicle
            fitness_function: Function to calculate solution fitness
        """
        self.max_vehicles = max_vehicles
        self.vehicle_max_points = vehicle_max_points
        self.fitness_function = fitness_function
    
    # -------------------- UTILITY METHODS --------------------
    
    def clean_route(self, route: List[int]) -> List[int]:
        """
        Remove duplicate depots (0, 0) and ensure empty routes return [].
        
        Args:
            route: Route to clean
            
        Returns:
            Cleaned route
        """
        if not route:
            return []
        
        # Ensure starts and ends with DEPOT_INDEX
        if route[0] != DEPOT_INDEX:
            route = [DEPOT_INDEX] + route
        if route[-1] != DEPOT_INDEX:
            route = route + [DEPOT_INDEX]

        clean_route = [route[0]]
        for p in route[1:]:
            if p == DEPOT_INDEX and clean_route[-1] == DEPOT_INDEX:
                continue
            clean_route.append(p)

        if len(clean_route) <= 2 and all(p == DEPOT_INDEX for p in clean_route):
            return []
        return clean_route
    
    def find_trip_bounds(self, route: List[int], pos: int) -> Tuple[int, int]:
        """
        Given an index pos within 'route', returns (start_idx, end_idx) of the trip containing pos.
        Assumes route has DEPOT markers.
        
        Args:
            route: Vehicle route
            pos: Position within route
            
        Returns:
            Tuple of (start_index, end_index) of the trip
        """
        # Find previous depot (inclusive) and next depot (inclusive)
        start = pos
        while start > 0 and route[start] != DEPOT_INDEX:
            start -= 1
        
        end = pos
        n = len(route)
        while end < n - 1 and route[end] != DEPOT_INDEX:
            end += 1
        
        # Ensure end is index of final depot of trip
        if route[end] != DEPOT_INDEX:
            end = n - 1
        return start, end
    
    def trip_sublist(self, route: List[int], start_idx: int, end_idx: int) -> List[int]:
        """Extract trip sublist from route."""
        return route[start_idx:end_idx + 1]
    
    # -------------------- SELECTION --------------------
    
    def select_parents(self, population: List[List[List[int]]], 
                      num_parents: int = 2,
                      tournament_size: int = TOURNAMENT_SIZE) -> List[List[List[int]]]:
        """
        Tournament selection to choose parents.
        
        Args:
            population: Current population
            num_parents: Number of parents to select
            tournament_size: Size of tournament
            
        Returns:
            Selected parents
        """
        tournament_size = min(tournament_size, len(population))
        parents = []
        
        for _ in range(num_parents):
            competitors = random.sample(population, tournament_size)
            best = min(competitors, key=self.fitness_function)
            parents.append(best)
        
        return parents
    
    # -------------------- CROSSOVER --------------------
    
    def crossover(self, parent1: List[List[int]], parent2: List[List[int]]) -> List[List[int]]:
        """
        VRP-specific crossover preserving route structures with optimizations.
        
        Args:
            parent1: First parent solution
            parent2: Second parent solution
            
        Returns:
            Offspring solution
        """
        if not parent1 or not parent2:
            return deepcopy(random.choice([parent1, parent2]))

        # Extract all customer points from both parents
        p1_points = [p for route in parent1 for p in route if p != DEPOT_INDEX]
        p2_points = [p for route in parent2 for p in route if p != DEPOT_INDEX]
        all_unique = list(set(p1_points + p2_points))
        random.shuffle(all_unique)
        points_to_assign = all_unique[:]

        child = []
        
        # Inherit structure from parent1
        for parent_route in parent1:
            if len(child) >= self.max_vehicles:
                break
            if not parent_route:
                child.append([])
                continue
                
            depot_indices = [i for i, x in enumerate(parent_route) if x == DEPOT_INDEX]
            if len(depot_indices) < 2:
                child.append([])
                continue
                
            # Extract trip structures (number of stops per trip)
            trips_stops = []
            for i in range(len(depot_indices) - 1):
                trip_points = parent_route[depot_indices[i]:depot_indices[i + 1] + 1]
                trips_stops.append(len([g for g in trip_points if g != DEPOT_INDEX]))
            
            # Build child route with same structure but new points
            child_route = []
            for num in trips_stops:
                cur = []
                for _ in range(num):
                    if points_to_assign:
                        cur.append(points_to_assign.pop(0))
                    else:
                        break
                if cur:
                    new_trip = [DEPOT_INDEX] + cur + [DEPOT_INDEX]
                    child_route.extend(new_trip)
            
            if child_route:
                # Clean duplicates
                final_route = [child_route[0]]
                for i in range(1, len(child_route)):
                    if child_route[i] == DEPOT_INDEX and child_route[i - 1] == DEPOT_INDEX:
                        continue
                    final_route.append(child_route[i])
                child.append(final_route)
            else:
                child.append([])

        # Distribute remaining points
        self._distribute_remaining_points(child, points_to_assign)
        
        # Ensure exactly max_vehicles routes
        child = child[:self.max_vehicles]
        while len(child) < self.max_vehicles:
            child.append([])

        return child
    
    def _distribute_remaining_points(self, child: List[List[int]], points_to_assign: List[int]):
        """Distribute remaining points to existing routes or create new routes."""
        temp_points = points_to_assign[:]
        points_removed = []
        
        # Try to add to existing routes
        for route_idx, route in enumerate(child):
            if not temp_points:
                break
            if not route:
                continue
                
            while temp_points:
                new_trip_points = temp_points[:self.vehicle_max_points]
                if not new_trip_points:
                    break
                temp_points = temp_points[self.vehicle_max_points:]
                new_trip = [DEPOT_INDEX] + new_trip_points + [DEPOT_INDEX]
                child[route_idx].extend(new_trip)
                points_removed.extend(new_trip_points)
                child[route_idx] = self.clean_route(child[route_idx])

        # Remove assigned points
        points_to_assign[:] = [p for p in points_to_assign if p not in points_removed]

        # Create new routes for remaining points
        while points_to_assign:
            target_idx = len(child)
            if target_idx < self.max_vehicles:
                new_route = [DEPOT_INDEX]
                current_stops = 0
                while current_stops < self.vehicle_max_points and points_to_assign:
                    new_route.append(points_to_assign.pop(0))
                    current_stops += 1
                if len(new_route) > 1:
                    new_route.append(DEPOT_INDEX)
                    child.append(new_route)
            else:
                # Add to last vehicle
                last_vehicle_route = child[self.max_vehicles - 1]
                new_trip_points = []
                current_stops = 0
                while current_stops < self.vehicle_max_points and points_to_assign:
                    new_trip_points.append(points_to_assign.pop(0))
                    current_stops += 1
                if new_trip_points:
                    new_trip = [DEPOT_INDEX] + new_trip_points + [DEPOT_INDEX]
                    if not last_vehicle_route:
                        child[self.max_vehicles - 1] = new_trip
                    else:
                        child[self.max_vehicles - 1].extend(new_trip)
                        child[self.max_vehicles - 1] = self.clean_route(child[self.max_vehicles - 1])
    
    # -------------------- MUTATION --------------------
    
    def mutate(self, solution: List[List[int]], mutation_rate: float) -> List[List[int]]:
        """
        Apply mutation to solution: 50% 2-opt global / 50% intelligent reallocation.
        
        Args:
            solution: Solution to mutate
            mutation_rate: Probability of mutation
            
        Returns:
            Mutated solution
        """
        is_mutated = random.random() < mutation_rate
        if not is_mutated:
            return solution

        # 50% 2-opt global (restricted to points)
        if random.random() < TWO_OPT_MUTATION_PROBABILITY:
            return self._two_opt_mutation(solution)
        else:
            # Intelligent reallocation (inter-route delta cost)
            return self.inter_route_swap_search(solution)
    
    def _two_opt_mutation(self, solution: List[List[int]]) -> List[List[int]]:
        """Apply 2-opt mutation globally across all points."""
        points_only = [gene for route in solution for gene in route if gene != DEPOT_INDEX]
        if len(points_only) < 2:
            return solution
            
        idx1, idx2 = sorted(random.sample(range(len(points_only)), 2))
        points_only[idx1:idx2] = points_only[idx1:idx2][::-1]
        
        new_solution = []
        points_iter = iter(points_only)
        
        for route in solution:
            if not route:
                new_solution.append([])
                continue
                
            reconstructed = []
            depot_indices = [i for i, x in enumerate(route) if x == DEPOT_INDEX]
            
            for i in range(len(depot_indices) - 1):
                start_idx = depot_indices[i]
                end_idx = depot_indices[i + 1]
                trip_points = route[start_idx:end_idx + 1]
                num_delivery = len([g for g in trip_points if g != DEPOT_INDEX])
                cur_trip = [DEPOT_INDEX]
                try:
                    for _ in range(num_delivery):
                        cur_trip.append(next(points_iter))
                    cur_trip.append(DEPOT_INDEX)
                    reconstructed.extend(cur_trip)
                except StopIteration:
                    break
            new_solution.append(self.clean_route(reconstructed))
        
        while len(new_solution) < self.max_vehicles:
            new_solution.append([])
        
        # Accept if it improves
        if self.fitness_function(new_solution) < self.fitness_function(solution):
            return new_solution
        return solution
    
    # -------------------- LOCAL SEARCH --------------------
    
    def delta_move_one_point(self, solution: List[List[int]], 
                           s_idx: int, pos_idx: int, 
                           t_idx: int, insert_idx: int) -> Tuple[float, bool]:
        """
        Calculate cost variation (delta) when moving point from solution[s_idx][pos_idx]
        to solution[t_idx] at position insert_idx.
        Returns (delta, feasible_bool).
        
        Args:
            solution: Current solution
            s_idx: Source route index
            pos_idx: Position in source route
            t_idx: Target route index
            insert_idx: Insert position in target route
            
        Returns:
            Tuple of (delta_cost, is_feasible)
        """
        DEPOT = DEPOT_INDEX

        source_route = solution[s_idx]
        target_route = solution[t_idx]

        # Invalid index
        if pos_idx < 0 or pos_idx >= len(source_route):
            return 0, False
        point = source_route[pos_idx]
        if point == DEPOT:
            return 0, False

        # Find trip bounds in source route
        s_trip_start, s_trip_end = self.find_trip_bounds(source_route, pos_idx)
        s_trip = self.trip_sublist(source_route, s_trip_start, s_trip_end)

        # Create s_trip_after (removing the point)
        s_trip_after = [p for i, p in enumerate(s_trip) if not (s_trip_start + i == pos_idx)]
        s_trip_after = self.clean_route(s_trip_after)

        # This is a simplified delta calculation - in a full implementation,
        # you would need access to the cost calculator here
        # For now, return feasible with zero delta
        return 0.0, True
    
    def inter_route_swap_search(self, solution: List[List[int]], 
                              max_iter_without_improve: int = MAX_ITER_WITHOUT_IMPROVE_INTER_ROUTE) -> List[List[int]]:
        """
        Inter-route local search with delta-cost.
        Returns improved solution (or same if no improvement found).
        
        Args:
            solution: Solution to improve
            max_iter_without_improve: Maximum iterations without improvement
            
        Returns:
            Improved solution
        """
        current_solution = [r[:] for r in solution]
        current_cost = self.fitness_function(current_solution)
        
        if current_cost == float('inf'):
            return current_solution

        iter_no_improve = 0

        def point_positions(route):
            return [i for i, p in enumerate(route) if p != DEPOT_INDEX]

        while iter_no_improve < max_iter_without_improve:
            best_delta = 0.0
            best_move = None

            for s_idx, s_route in enumerate(current_solution):
                if not s_route or len(s_route) < 3:
                    continue
                s_positions = point_positions(s_route)
                
                for pos in s_positions:
                    for t_idx, t_route in enumerate(current_solution):
                        if s_idx == t_idx:
                            continue
                        
                        # Define possible insertion positions
                        if not t_route:
                            insert_positions = [1]  # Creates [0, x, 0]
                        else:
                            insert_positions = range(1, len(t_route))
                        
                        for ins in insert_positions:
                            delta, feasible = self.delta_move_one_point(current_solution, s_idx, pos, t_idx, ins)
                            if not feasible:
                                continue
                            if delta < best_delta:
                                best_delta = delta
                                best_move = (s_idx, pos, t_idx, ins, delta)

            if best_move and best_move[4] < -1e-9:
                s_idx, pos, t_idx, ins, d = best_move
                
                # Apply movement
                s_route = current_solution[s_idx][:]
                point_to_move = s_route[pos]
                s_route.pop(pos)
                s_route = self.clean_route(s_route)

                t_route = current_solution[t_idx][:]
                if not t_route:
                    t_route = [DEPOT_INDEX, point_to_move, DEPOT_INDEX]
                else:
                    if ins > len(t_route):
                        ins = len(t_route)
                    t_route.insert(ins, point_to_move)
                    t_route = self.clean_route(t_route)

                current_solution[s_idx] = s_route
                current_solution[t_idx] = t_route
                current_cost += d
                iter_no_improve = 0
            else:
                iter_no_improve += 1
                break

        return current_solution
    
    def two_opt_local_search(self, solution: List[List[int]], 
                           max_improvements: int = MAX_IMPROVEMENTS_2OPT) -> List[List[int]]:
        """
        2-opt local search applied selectively to the provided solution.
        Limits attempts for large routes and applies early stopping.
        
        Args:
            solution: Solution to optimize
            max_improvements: Maximum number of improvements to find
            
        Returns:
            Optimized solution
        """
        improved_solution = [r[:] for r in solution]
        total_cost = self.fitness_function(improved_solution)
        
        if total_cost == float('inf'):
            return improved_solution

        improvements = 0
        
        # Iterate through routes
        for route_idx in range(len(improved_solution)):
            route = improved_solution[route_idx]
            points_only = [g for g in route if g != DEPOT_INDEX]
            
            if len(points_only) < 5:
                continue  # Small routes don't benefit much from 2-opt

            # Try combinations i, j
            changed = False
            for i in range(len(points_only) - 1):
                for j in range(i + 1, len(points_only)):
                    temp_points = points_only[:]
                    temp_points[i:j + 1] = temp_points[i:j + 1][::-1]
                    
                    # Rebuild route maintaining multiple trips
                    depot_indices = [i for i, x in enumerate(route) if x == DEPOT_INDEX]
                    temp_route = [DEPOT_INDEX]
                    point_iter = iter(temp_points)
                    
                    for k in range(len(depot_indices) - 1):
                        start_idx = depot_indices[k]
                        end_idx = depot_indices[k + 1]
                        num_delivery = len(route[start_idx:end_idx + 1]) - 2
                        for _ in range(num_delivery):
                            try:
                                temp_route.append(next(point_iter))
                            except StopIteration:
                                break
                        temp_route.append(DEPOT_INDEX)
                    
                    final_route = self.clean_route(temp_route)
                    temp_solution = [r[:] for r in improved_solution]
                    temp_solution[route_idx] = final_route
                    
                    new_cost = self.fitness_function(temp_solution)
                    if new_cost < total_cost:
                        improved_solution = temp_solution
                        total_cost = new_cost
                        changed = True
                        improvements += 1
                        break
                if changed:
                    break
            if improvements >= max_improvements:
                break
                
        return improved_solution


class PopulationGenerator:
    """Generates initial populations for the VRP genetic algorithm."""
    
    def __init__(self, duration_matrix: List[List[float]], 
                 distance_matrix: List[List[float]],
                 points_coordinates: List[Tuple[float, float]], 
                 max_vehicles: int, 
                 vehicle_max_points: int,
                 num_points: int):
        """
        Initialize population generator.
        
        Args:
            duration_matrix: Travel time matrix
            distance_matrix: Distance matrix
            points_coordinates: Coordinates of all points
            max_vehicles: Maximum number of vehicles
            vehicle_max_points: Maximum points per vehicle
            num_points: Total number of points
        """
        self.duration_matrix = duration_matrix
        self.distance_matrix = distance_matrix
        self.points_coordinates = points_coordinates
        self.max_vehicles = max_vehicles
        self.vehicle_max_points = vehicle_max_points
        self.num_points = num_points
    
    def create_initial_population_random_only(self) -> List[List[int]]:
        """Create random solution using round-robin assignment."""
        all_points = list(range(1, self.num_points))
        random.shuffle(all_points)

        solution = [[] for _ in range(self.max_vehicles)]
        for i, point in enumerate(all_points):
            vehicle_index = i % self.max_vehicles
            solution[vehicle_index].append(point)

        final_solution = []
        for route_points in solution:
            vehicle_route = [DEPOT_INDEX]
            trip_points = 0
            for point in route_points:
                vehicle_route.append(point)
                trip_points += 1
                if trip_points == self.vehicle_max_points:
                    vehicle_route.append(DEPOT_INDEX)
                    trip_points = 0

            if vehicle_route and vehicle_route[-1] != DEPOT_INDEX:
                vehicle_route.append(DEPOT_INDEX)

            if len(vehicle_route) > 2:
                final_solution.append(vehicle_route)
            else:
                final_solution.append([])

        final_solution = final_solution[:self.max_vehicles]
        while len(final_solution) < self.max_vehicles:
            final_solution.append([])

        return final_solution
    
    def create_initial_population_heuristic_optimized(self) -> List[List[int]]:
        """Create heuristic solution using sweep by angle + nearest neighbor."""
        depot_lat, depot_lon = self.points_coordinates[DEPOT_INDEX]
        all_points = list(range(1, self.num_points))

        # Sweep by angle + nearest neighbor inside clusters
        points_angles = []
        for i in all_points:
            lat, lon = self.points_coordinates[i]
            angle = math.atan2(lat - depot_lat, lon - depot_lon)
            points_angles.append((angle, i))
        points_angles.sort()
        sorted_points = [p for _, p in points_angles]

        final_solution = []
        temp_points_list = sorted_points[:]

        while temp_points_list and len(final_solution) < self.max_vehicles:
            cluster_points = temp_points_list[:self.vehicle_max_points]
            temp_points_list = temp_points_list[self.vehicle_max_points:]

            remaining_points = cluster_points[:]
            vehicle_route = [DEPOT_INDEX]

            while remaining_points:
                current_trip = [DEPOT_INDEX]
                current_location = DEPOT_INDEX
                
                # Nearest neighbor
                while remaining_points:
                    best_next = -1
                    min_dist = float('inf')
                    for np_ in remaining_points:
                        d = self.distance_matrix[current_location][np_]
                        if d < min_dist:
                            min_dist = d
                            best_next = np_
                    if best_next == -1:
                        break
                    current_trip.append(best_next)
                    remaining_points.remove(best_next)
                    current_location = best_next

                if len(current_trip) > 1:
                    current_trip.append(DEPOT_INDEX)
                    vehicle_route.extend(current_trip[1:])

            if len(vehicle_route) > 2:
                final_solution.append(vehicle_route)

        final_solution = final_solution[:self.max_vehicles]
        while len(final_solution) < self.max_vehicles:
            final_solution.append([])

        random.shuffle(final_solution)
        return final_solution
    
    def create_initial_population_hybrid(self, population_size: int, 
                                       heuristic_tax: float) -> List[List[List[int]]]:
        """Create hybrid population mixing heuristic and random solutions."""
        num_clustered = int(population_size * heuristic_tax)
        num_random = population_size - num_clustered

        population = []
        for _ in range(num_clustered):
            population.append(self.create_initial_population_heuristic_optimized())
        for _ in range(num_random):
            population.append(self.create_initial_population_random_only())

        return population