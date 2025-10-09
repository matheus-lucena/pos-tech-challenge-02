"""
Main VRP Genetic Algorithm class with clean architecture.
This module contains the main algorithm orchestrator using the separated components.
"""

import time
import logging
from typing import List, Tuple, Callable, Optional


from vrp.config import MUTATION_RATE_INCREASE_FACTOR, MAX_ITER_WITHOUT_IMPROVE_INTER_ROUTE, MAX_IMPROVEMENTS_2OPT, MAX_MUTATION_RATE,DEPOT_INDEX, TWO_OPT_FREQUENCY, MAX_IMPROVEMENTS_2OPT, DEFAULT_MUTATION_RATE, COUNT_GENERATIONS_WITHOUT_IMPROVEMENT, DEFAULT_TIME_WEIGHT, DEFAULT_DISTANCE_WEIGHT, COUNT_GENERATIONS_WITHOUT_IMPROVEMENT_FOR_MUTATION
from vrp.cost_and_workers import CostCalculator, ParallelFitnessEvaluator
from vrp.vrp_operators import VRPOperators, PopulationGenerator


class VRPGeneticAlgorithm:
    """
    Optimized VRP Genetic Algorithm with clean architecture and parallel processing.
    """
    
    def __init__(self, 
                 duration_matrix: List[List[float]], 
                 distance_matrix: List[List[float]], 
                 points: List[Tuple[float, float]],
                 max_vehicles: int, 
                 vehicle_max_points: int,
                 generations: int, 
                 population_size: int, 
                 population_heuristic_tax: float,
                 max_trip_duration: int, 
                 max_trip_distance: int, 
                 time_to_stop: int,
                 mutation_rate: float = DEFAULT_MUTATION_RATE, 
                 max_no_improvement: int = COUNT_GENERATIONS_WITHOUT_IMPROVEMENT, 
                 time_weight: float = DEFAULT_TIME_WEIGHT, 
                 distance_weight: float = DEFAULT_DISTANCE_WEIGHT):
        """
        Initialize VRP Genetic Algorithm.
        
        Args:
            duration_matrix: Travel time matrix between points
            distance_matrix: Distance matrix between points
            points: List of (latitude, longitude) coordinates
            max_vehicles: Maximum number of vehicles
            vehicle_max_points: Maximum points per vehicle
            generations: Number of generations to run
            population_size: Size of population
            population_heuristic_tax: Proportion of heuristic solutions in initial population
            max_trip_duration: Maximum duration per trip
            max_trip_distance: Maximum distance per trip
            time_to_stop: Service time at each customer
            mutation_rate: Probability of mutation
            max_no_improvement: Generations without improvement before restart
            time_weight: Weight for time component in fitness
            distance_weight: Weight for distance component in fitness
        """
        # Algorithm parameters
        self.max_vehicles = max_vehicles
        self.vehicle_max_points = vehicle_max_points
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.population_size = population_size
        self.heuristic_tax = population_heuristic_tax
        self.max_trip_duration = max_trip_duration
        self.max_trip_distance = max_trip_distance
        self.time_to_stop = time_to_stop
        self.max_no_improvement = max_no_improvement

        # Problem data
        self.duration_matrix = duration_matrix
        self.distance_matrix = distance_matrix
        self.points_coordinates = points
        self.num_points = len(points)
        self.time_weight = time_weight
        self.distance_weight = distance_weight

        # Initialize components
        self._initialize_components()
        
        # Initialize fitness cache for performance
        self.fitness_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Create initial population
        self.population = self.population_generator.create_initial_population_hybrid(
            self.population_size, self.heuristic_tax
        )

    def _initialize_components(self):
        """Initialize all algorithm components."""
        # Cost calculator for local calculations
        self.cost_calculator = CostCalculator(
            self.duration_matrix, self.distance_matrix,
            self.max_trip_duration, self.max_trip_distance, self.time_to_stop,
            self.time_weight, self.distance_weight
        )
        
        # Parallel fitness evaluator
        self.parallel_evaluator = ParallelFitnessEvaluator(
            self.duration_matrix, self.distance_matrix, self.num_points,
            self.vehicle_max_points, self.max_trip_duration, self.max_trip_distance,
            self.time_to_stop, self.time_weight, self.distance_weight
        )
        
        # Population generator
        self.population_generator = PopulationGenerator(
            self.duration_matrix, self.distance_matrix, self.points_coordinates,
            self.max_vehicles, self.vehicle_max_points, self.num_points
        )
        
        # VRP operators
        self.operators = VRPOperators(
            self.max_vehicles, self.vehicle_max_points, self.calculate_fitness
        )

    def calculate_fitness(self, solution: List[List[int]]) -> float:
        """
        Calculate fitness of a solution using the local cost calculator with caching.
        
        Args:
            solution: VRP solution to evaluate
            
        Returns:
            Fitness score (lower is better)
        """
        # Create hashable key for caching
        solution_key = tuple(tuple(route) for route in solution)
        
        # Check cache first
        if solution_key in self.fitness_cache:
            self.cache_hits += 1
            return self.fitness_cache[solution_key]
        
        # Calculate fitness if not in cache
        self.cache_misses += 1
        fitness = self.cost_calculator.calculate_fitness(solution, self.max_vehicles, self.num_points)
        
        # Store in cache (with size limit to prevent memory issues)
        if len(self.fitness_cache) < 30000:  # Limit cache size
            self.fitness_cache[solution_key] = fitness
        
        return fitness

    def run(self, epoch_callback: Optional[Callable] = None) -> Tuple[List[List[int]], float]:
        """
        Execute the genetic algorithm main loop.
        
        Args:
            epoch_callback: Optional callback function called after each generation
            
        Returns:
            Tuple of (best_solution, best_cost)
        """
        logging.info("Starting VRP Genetic Algorithm execution")
        
        # Calculate chunk size for parallel processing
        chunksize = (self.population_size // self.parallel_evaluator.cpu_count 
                    if self.population_size > self.parallel_evaluator.cpu_count else 1)

        # Initialize fitness cache
        fitness_cache = [self.calculate_fitness(sol) for sol in self.population]
        best_idx = min(range(len(fitness_cache)), key=lambda i: fitness_cache[i])
        best_solution = self.population[best_idx]
        best_cost = fitness_cache[best_idx]

        # Algorithm state
        count_generations_without_improvement = 0
        count_generations_without_improvement_for_mutation = 0
        mutation_rate = self.mutation_rate

        # Main evolution loop
        for generation in range(self.generations):
            start_gen = time.time()
            new_population = []

            # Population restart on stagnation
            if count_generations_without_improvement >= self.max_no_improvement:
                logging.info(f"Generation {generation + 1}: Restarting population after {self.max_no_improvement} generations without improvement")
                self.population = self.population_generator.create_initial_population_hybrid(
                    self.population_size, self.heuristic_tax
                )
                fitness_cache = [self.calculate_fitness(sol) for sol in self.population]
                count_generations_without_improvement = 0
                mutation_rate = self.mutation_rate

            # Adaptive mutation rate increase
            if count_generations_without_improvement_for_mutation >= COUNT_GENERATIONS_WITHOUT_IMPROVEMENT_FOR_MUTATION:
                logging.info(f"Generation {generation + 1}: Increasing mutation rate")
                mutation_rate = min(MAX_MUTATION_RATE, mutation_rate * MUTATION_RATE_INCREASE_FACTOR)
                count_generations_without_improvement_for_mutation = 0

            # Apply local search optimizations
            self._apply_local_optimizations(generation, best_solution, best_cost, 
                                          count_generations_without_improvement,
                                          count_generations_without_improvement_for_mutation)

            # Selection and reproduction with elitism
            best_gen_idx = min(range(len(fitness_cache)), key=lambda i: fitness_cache[i])
            best_of_gen = self.population[best_gen_idx]
            new_population.append(best_of_gen)  # Elitism

            # Generate rest of population
            while len(new_population) < self.population_size:
                parents = self.operators.select_parents(self.population)
                child = self.operators.crossover(parents[0], parents[1])
                mutated = self.operators.mutate(child, mutation_rate)
                new_population.append(mutated)

            # Parallel fitness evaluation
            new_fitness_cache = self.parallel_evaluator.evaluate_population(new_population, chunksize)

            # Update population and fitness cache
            self.population = new_population
            fitness_cache = new_fitness_cache

            # Find current best
            current_best_idx = min(range(len(fitness_cache)), key=lambda i: fitness_cache[i])
            current_best = self.population[current_best_idx]
            current_best_cost = fitness_cache[current_best_idx]

            logging.info(f'Generation {generation + 1} - Best in population: {current_best_cost:.4f}')

            # Update global best
            if current_best_cost < best_cost:
                best_solution = current_best
                best_cost = current_best_cost
                count_generations_without_improvement = 0
                count_generations_without_improvement_for_mutation = 0
            else:
                count_generations_without_improvement += 1
                count_generations_without_improvement_for_mutation += 1

            generation_time = time.time() - start_gen
            logging.info(f'Generation {generation + 1} - Global best: {best_cost:.4f} (time: {generation_time:.2f}s)')

            # Call progress callback
            if epoch_callback:
                self._call_progress_callback(epoch_callback, generation + 1, best_solution)

        # Log cache statistics
        total_evaluations = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / total_evaluations * 100) if total_evaluations > 0 else 0
        
        logging.info(f"Algorithm completed. Final best cost: {best_cost:.4f}")
        logging.info(f"Fitness cache statistics: {self.cache_hits} hits, {self.cache_misses} misses ({cache_hit_rate:.1f}% hit rate)")
        logging.info(f"Total fitness evaluations reduced from {total_evaluations} to {self.cache_misses} ({total_evaluations - self.cache_misses} saved)")
        
        return best_solution, best_cost

    def _apply_local_optimizations(self, generation: int, best_solution: List[List[int]], 
                                 best_cost: float, count_no_improve: int, 
                                 count_no_improve_mut: int) -> Tuple[List[List[int]], float]:
        """Apply local search optimizations periodically."""
        # Inter-route local search using delta-cost
        if generation % COUNT_GENERATIONS_WITHOUT_IMPROVEMENT_FOR_MUTATION == 0 and best_cost != float('inf'):
            logging.debug(f"Generation {generation + 1}: Applying inter-route local search")
            inter_opt = self.operators.inter_route_swap_search(
                best_solution, max_iter_without_improve=MAX_ITER_WITHOUT_IMPROVE_INTER_ROUTE
            )
            inter_cost = self.calculate_fitness(inter_opt)
            
            if inter_cost < best_cost:
                best_cost = inter_cost
                best_solution = inter_opt
                count_no_improve = 0
                count_no_improve_mut = 0
                logging.info(f"Generation {generation + 1}: Inter-route search improved to {best_cost:.4f}")

        # 2-opt applied only to best_solution at TWO_OPT_FREQUENCY
        if (generation % TWO_OPT_FREQUENCY == 0 and 
            best_cost != float('inf') and 
            any(len([p for p in r if p != DEPOT_INDEX]) >= 6 for r in best_solution)):
            
            logging.debug(f"Generation {generation + 1}: Applying 2-opt to best solution")
            local_opt = self.operators.two_opt_local_search(
                best_solution, max_improvements=MAX_IMPROVEMENTS_2OPT
            )
            local_cost = self.calculate_fitness(local_opt)
            
            if local_cost < best_cost:
                best_cost = local_cost
                best_solution = local_opt
                count_no_improve = 0
                logging.info(f"Generation {generation + 1}: 2-opt improved to {best_cost:.4f}")

        return best_solution, best_cost

    def _call_progress_callback(self, callback: Callable, epoch: int, best_solution: List[List[int]]):
        """Call progress callback with vehicle information."""
        try:
            vehicle_points = []
            for i, route in enumerate(best_solution):
                if route:
                    trip_cost, trip_duration, trip_distance = self.cost_calculator.trip_cost_local(route)
                    vehicle_points.append({
                        'vehicle_id': i + 1,
                        'points': len([p for p in route if p != DEPOT_INDEX]),
                        'route': route,
                        'distance': round(trip_distance / 1000, 2) if trip_distance != float('inf') else None,
                        'duration': round(trip_duration / 60, 1) if trip_duration != float('inf') else None,
                        'cost': round(trip_cost, 6) if trip_cost != float('inf') else None
                    })
            
            callback(epoch=epoch, vehicles=len(vehicle_points), vehicle_data=vehicle_points)
        except Exception as e:
            logging.warning(f"Progress callback failed: {e}")

    def get_solution_statistics(self, solution: List[List[int]] = None) -> dict:
        """
        Get comprehensive statistics about a solution.
        
        Args:
            solution: Solution to analyze (uses best solution if None)
            
        Returns:
            Dictionary with solution statistics
        """
        if solution is None:
            if not hasattr(self, 'best_solution'):
                return {"error": "No solution available"}
            solution = self.best_solution

        stats = {
            "fitness_score": self.calculate_fitness(solution),
            "total_routes": len(solution),
            "active_vehicles": len([r for r in solution if r]),
            "total_customer_points": sum(len([p for p in route if p != DEPOT_INDEX]) for route in solution),
            "routes": []
        }

        for i, route in enumerate(solution):
            if route:
                customer_points = len([p for p in route if p != DEPOT_INDEX])
                trip_cost, trip_duration, trip_distance = self.cost_calculator.trip_cost_local(route)
                
                route_stats = {
                    "vehicle_id": i + 1,
                    "customer_points": customer_points,
                    "route": route,
                    "cost": trip_cost if trip_cost != float('inf') else None,
                    "duration_minutes": round(trip_duration / 60, 1) if trip_duration != float('inf') else None,
                    "distance_km": round(trip_distance / 1000, 2) if trip_distance != float('inf') else None
                }
                stats["routes"].append(route_stats)

        return stats

