"""
Cost calculation and worker functions for parallel processing in the VRP Genetic Algorithm.
This module handles fitness calculations and multiprocessing worker functions.
"""

import multiprocessing
from typing import List, Tuple, Set
from vrp.config import DEPOT_INDEX, TIME_DEPOT_STOP, DURATION_PENALTY_THRESHOLD, DISTANCE_PENALTY_THRESHOLD, PENALTY_MULTIPLIER


# Global worker variables (initialized via initializer)
_worker_duration_matrix = None
_worker_distance_matrix = None
_worker_num_points = None
_worker_vehicle_max_points = None
_worker_max_trip_duration = None
_worker_max_trip_distance = None
_worker_time_to_stop = None
_worker_time_weight = None
_worker_distance_weight = None


def _init_worker(duration_matrix: List[List[float]], 
                distance_matrix: List[List[float]], 
                num_points: int,
                vehicle_max_points: int, 
                max_trip_duration: int, 
                max_trip_distance: int,
                time_to_stop: int, 
                time_weight: float, 
                distance_weight: float):
    """
    Initializer for multiprocessing workers: loads matrices globally in the worker.
    
    Args:
        duration_matrix: Travel time matrix between points
        distance_matrix: Distance matrix between points
        num_points: Total number of points including depot
        vehicle_max_points: Maximum points per vehicle
        max_trip_duration: Maximum duration per trip
        max_trip_distance: Maximum distance per trip
        time_to_stop: Service time at each customer
        time_weight: Weight for time component in fitness
        distance_weight: Weight for distance component in fitness
    """
    global _worker_duration_matrix, _worker_distance_matrix, _worker_num_points
    global _worker_vehicle_max_points, _worker_max_trip_duration, _worker_max_trip_distance
    global _worker_time_to_stop, _worker_time_weight, _worker_distance_weight

    _worker_duration_matrix = duration_matrix
    _worker_distance_matrix = distance_matrix
    _worker_num_points = num_points
    _worker_vehicle_max_points = vehicle_max_points
    _worker_max_trip_duration = max_trip_duration
    _worker_max_trip_distance = max_trip_distance
    _worker_time_to_stop = time_to_stop
    _worker_time_weight = time_weight
    _worker_distance_weight = distance_weight


def fitness_worker(solution: List[List[int]]) -> float:
    """
    Streamlined fitness function for workers: calculates total cost of 'solution'
    using global matrices already loaded in the worker process.
    Returns float('inf') for infeasible solutions.
    
    Args:
        solution: VRP solution as list of vehicle routes
        
    Returns:
        Fitness score (lower is better, inf if infeasible)
    """
    DEPOT = DEPOT_INDEX
    
    # Sanity checks
    if not solution:
        return float('inf')

    total_solution_cost = 0.0
    vehicles_used = len([r for r in solution if r])
    
    # Protection against absurd values (should not occur)
    if vehicles_used > 0 and vehicles_used > 10000:
        return float('inf')

    visited_points: Set[int] = set()
    all_delivery_points: List[int] = []

    for vehicle_route in solution:
        if not vehicle_route:
            continue

        if vehicle_route[0] != DEPOT or vehicle_route[-1] != DEPOT:
            return float('inf')

        depot_indices = [i for i, x in enumerate(vehicle_route) if x == DEPOT]
        if len(depot_indices) < 2:
            return float('inf')

        # For each trip within the route (there may be multiple DEPOT...DEPOT segments)
        for i in range(len(depot_indices) - 1):
            start_idx = depot_indices[i]
            end_idx = depot_indices[i + 1]
            trip = vehicle_route[start_idx:end_idx + 1]
            
            if len(trip) <= 2:
                # Route without deliveries in this trip
                continue

            trip_cost, _, _ = _trip_cost_worker(trip)
            if trip_cost == float('inf'):
                return float('inf')

            total_solution_cost += trip_cost

            deliveries = [p for p in trip if p != DEPOT]
            visited_points.update(deliveries)
            all_delivery_points.extend(deliveries)

    # Check for point duplicates
    if len(all_delivery_points) != len(visited_points):
        return float('inf')

    # Check coverage constraint
    required = set(range(1, _worker_num_points))
    if visited_points != required:
        return float('inf')

    return total_solution_cost


def _trip_cost_worker(trip_points: List[int]) -> Tuple[float, int, int]:
    """
    Calculate cost (normalized) of a trip [0, p1, ..., 0] using global matrices in worker.
    Returns (trip_cost, duration_seconds, distance_units).
    
    Args:
        trip_points: List of points in the trip including start and end depot
        
    Returns:
        Tuple of (cost, duration, distance)
    """
    DEPOT = DEPOT_INDEX
    
    if len(trip_points) < 2 or trip_points[0] != DEPOT or trip_points[-1] != DEPOT:
        return float('inf'), 0, 0

    cur_duration = TIME_DEPOT_STOP  # Initial wait at depot
    cur_distance = 0
    last = trip_points[0]

    for p in trip_points[1:]:
        # Look up in global matrices (assuming correct indexing)
        dtime = _worker_duration_matrix[last][p]
        ddist = _worker_distance_matrix[last][p]
        cur_duration += dtime
        cur_distance += ddist

        if p != DEPOT:
            cur_duration += _worker_time_to_stop
        elif p == DEPOT and last != DEPOT:
            cur_duration += TIME_DEPOT_STOP

        last = p

    # Limit validations
    if cur_duration > _worker_max_trip_duration or cur_distance > _worker_max_trip_distance:
        return float('inf'), cur_duration, cur_distance

    # Calculate normalized ratios
    duration_ratio = cur_duration / _worker_max_trip_duration
    distance_ratio = cur_distance / _worker_max_trip_distance
    
    # Apply penalty for high utilization
    penalty = 1.0
    if duration_ratio > DURATION_PENALTY_THRESHOLD or distance_ratio > DISTANCE_PENALTY_THRESHOLD:
        max_ratio = max(duration_ratio, distance_ratio)
        penalty += PENALTY_MULTIPLIER * ((max_ratio - DURATION_PENALTY_THRESHOLD) / (1.0 - DURATION_PENALTY_THRESHOLD)) ** 2

    trip_cost = penalty * (_worker_time_weight * duration_ratio + _worker_distance_weight * distance_ratio)
    return trip_cost, cur_duration, cur_distance


class CostCalculator:
    """Local cost calculator for the main process (non-worker calculations)."""
    
    def __init__(self, duration_matrix: List[List[float]], 
                 distance_matrix: List[List[float]],
                 max_trip_duration: int, 
                 max_trip_distance: int,
                 time_to_stop: int,
                 time_weight: float = 0.5, 
                 distance_weight: float = 0.5):
        """
        Initialize cost calculator.
        
        Args:
            duration_matrix: Travel time matrix
            distance_matrix: Distance matrix
            max_trip_duration: Maximum trip duration
            max_trip_distance: Maximum trip distance
            time_to_stop: Service time at customers
            time_weight: Weight for time component
            distance_weight: Weight for distance component
        """
        self.duration_matrix = duration_matrix
        self.distance_matrix = distance_matrix
        self.max_trip_duration = max_trip_duration
        self.max_trip_distance = max_trip_distance
        self.time_to_stop = time_to_stop
        self.time_weight = time_weight
        self.distance_weight = distance_weight
    
    def trip_cost_local(self, trip_points: List[int]) -> Tuple[float, int, int]:
        """
        Local version of cost calculation similar to worker (for use in main process).
        Returns (trip_cost, duration, distance).
        
        Args:
            trip_points: List of points in the trip
            
        Returns:
            Tuple of (cost, duration, distance)
        """
        DEPOT = DEPOT_INDEX
        
        if len(trip_points) < 2 or trip_points[0] != DEPOT or trip_points[-1] != DEPOT:
            return float('inf'), 0, 0

        cur_duration = TIME_DEPOT_STOP
        cur_distance = 0
        last = trip_points[0]
        
        for p in trip_points[1:]:
            dtime = self.duration_matrix[last][p]
            ddist = self.distance_matrix[last][p]
            cur_duration += dtime
            cur_distance += ddist

            if p != DEPOT:
                cur_duration += self.time_to_stop
            elif p == DEPOT and last != DEPOT:
                cur_duration += TIME_DEPOT_STOP

            last = p

        if cur_duration > self.max_trip_duration or cur_distance > self.max_trip_distance:
            return float('inf'), cur_duration, cur_distance

        duration_ratio = cur_duration / self.max_trip_duration
        distance_ratio = cur_distance / self.max_trip_distance
        
        penalty = 1.0
        if duration_ratio > DURATION_PENALTY_THRESHOLD or distance_ratio > DISTANCE_PENALTY_THRESHOLD:
            max_ratio = max(duration_ratio, distance_ratio)
            penalty += PENALTY_MULTIPLIER * ((max_ratio - DURATION_PENALTY_THRESHOLD) / (1.0 - DURATION_PENALTY_THRESHOLD)) ** 2

        trip_cost = penalty * (self.time_weight * duration_ratio + self.distance_weight * distance_ratio)
        return trip_cost, cur_duration, cur_distance
    
    def calculate_fitness(self, solution: List[List[int]], max_vehicles: int, num_points: int) -> float:
        """
        Optimized fitness calculation with reduced complexity.
        
        Args:
            solution: VRP solution to evaluate
            max_vehicles: Maximum number of vehicles
            num_points: Total number of points
            
        Returns:
            Fitness score (lower is better, inf if infeasible)
        """
        total_solution_cost = 0.0
        visited_points = set()
        point_count = 0
        
        # Count active vehicles in single pass
        active_vehicles = 0
        for vehicle_route in solution:
            if vehicle_route:
                active_vehicles += 1
                
        if active_vehicles > max_vehicles:
            return float('inf')

        # Process each vehicle route
        for vehicle_route in solution:
            if not vehicle_route:
                continue

            # Quick validation
            if vehicle_route[0] != DEPOT_INDEX or vehicle_route[-1] != DEPOT_INDEX:
                return float('inf')

            # Find depot positions in single pass
            depot_indices = []
            for i, x in enumerate(vehicle_route):
                if x == DEPOT_INDEX:
                    depot_indices.append(i)
            
            if len(depot_indices) < 2:
                return float('inf')

            # Process trips with optimized loop
            for i in range(len(depot_indices) - 1):
                start_idx = depot_indices[i]
                end_idx = depot_indices[i + 1]
                trip = vehicle_route[start_idx:end_idx + 1]
                
                if len(trip) <= 2:
                    continue
                    
                # Fast trip cost calculation
                trip_cost = self._fast_trip_cost(trip)
                if trip_cost == float('inf'):
                    return float('inf')
                    
                total_solution_cost += trip_cost
                
                # Count and track points efficiently
                for p in trip[1:-1]:  # Skip depot at start and end
                    if p in visited_points:
                        return float('inf')  # Duplicate point
                    visited_points.add(p)
                    point_count += 1

        # Fast coverage check
        if point_count != num_points - 1:  # -1 for depot
            return float('inf')

        return total_solution_cost

    def _fast_trip_cost(self, trip_points: List[int]) -> float:
        """
        Optimized trip cost calculation with minimal operations.
        
        Args:
            trip_points: Points in the trip
            
        Returns:
            Trip cost or infinity if infeasible
        """
        if len(trip_points) < 2:
            return float('inf')

        cur_duration = TIME_DEPOT_STOP
        cur_distance = 0
        last = trip_points[0]
        
        # Single loop with optimized operations
        for p in trip_points[1:]:
            cur_duration += self.duration_matrix[last][p]
            cur_distance += self.distance_matrix[last][p]

            if p != DEPOT_INDEX:
                cur_duration += self.time_to_stop
            elif last != DEPOT_INDEX:  # Arriving at depot from customer
                cur_duration += TIME_DEPOT_STOP

            last = p

        # Quick feasibility check
        if cur_duration > self.max_trip_duration or cur_distance > self.max_trip_distance:
            return float('inf')

        # Simplified cost calculation
        duration_ratio = cur_duration / self.max_trip_duration
        distance_ratio = cur_distance / self.max_trip_distance
        
        # Fast penalty calculation
        penalty = 1.0
        max_ratio = max(duration_ratio, distance_ratio)
        if max_ratio > DURATION_PENALTY_THRESHOLD:
            excess = (max_ratio - DURATION_PENALTY_THRESHOLD) / (1.0 - DURATION_PENALTY_THRESHOLD)
            penalty += PENALTY_MULTIPLIER * excess * excess

        return penalty * (self.time_weight * duration_ratio + self.distance_weight * distance_ratio)


class ParallelFitnessEvaluator:
    """Handles parallel fitness evaluation using multiprocessing."""
    
    def __init__(self, duration_matrix: List[List[float]], 
                 distance_matrix: List[List[float]],
                 num_points: int,
                 vehicle_max_points: int,
                 max_trip_duration: int, 
                 max_trip_distance: int,
                 time_to_stop: int,
                 time_weight: float = 0.5, 
                 distance_weight: float = 0.5):
        """
        Initialize parallel fitness evaluator.
        
        Args:
            duration_matrix: Travel time matrix
            distance_matrix: Distance matrix
            num_points: Total number of points
            vehicle_max_points: Maximum points per vehicle
            max_trip_duration: Maximum trip duration
            max_trip_distance: Maximum trip distance
            time_to_stop: Service time at customers
            time_weight: Weight for time component
            distance_weight: Weight for distance component
        """
        self.pool_init_args = (
            duration_matrix, distance_matrix, num_points,
            vehicle_max_points, max_trip_duration, max_trip_distance,
            time_to_stop, time_weight, distance_weight
        )
        self.cpu_count = max(1, multiprocessing.cpu_count())
    
    def evaluate_population(self, population: List[List[List[int]]], 
                          chunksize: int = None) -> List[float]:
        """
        Evaluate fitness of entire population in parallel.
        
        Args:
            population: Population to evaluate
            chunksize: Chunk size for parallel processing
            
        Returns:
            List of fitness scores
        """
        if chunksize is None:
            chunksize = len(population) // self.cpu_count if len(population) > self.cpu_count else 1
        
        with multiprocessing.Pool(processes=self.cpu_count, 
                                initializer=_init_worker, 
                                initargs=self.pool_init_args) as pool:
            fitness_scores = pool.map(fitness_worker, population, chunksize=chunksize)
        
        return fitness_scores