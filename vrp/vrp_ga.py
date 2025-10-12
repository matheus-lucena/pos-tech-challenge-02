from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import threading
import time
import logging
import hashlib

from typing import List, Tuple, Callable, Optional

from vrp.config import MUTATION_RATE_INCREASE_FACTOR, MAX_ITER_WITHOUT_IMPROVE_INTER_ROUTE, MAX_IMPROVEMENTS_2OPT, MAX_MUTATION_RATE,DEPOT_INDEX, TWO_OPT_FREQUENCY, MAX_IMPROVEMENTS_2OPT, DEFAULT_MUTATION_RATE, COUNT_GENERATIONS_WITHOUT_IMPROVEMENT, DEFAULT_TIME_WEIGHT, DEFAULT_DISTANCE_WEIGHT, COUNT_GENERATIONS_WITHOUT_IMPROVEMENT_FOR_MUTATION
from vrp.cost_and_workers import CostCalculator, ParallelFitnessEvaluator
from vrp.vrp_operators import VRPOperators, PopulationGenerator


class VRPGeneticAlgorithm:
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

        self.duration_matrix = duration_matrix
        self.distance_matrix = distance_matrix
        self.points_coordinates = points
        self.num_points = len(points)
        self.time_weight = time_weight
        self.distance_weight = distance_weight

        self._initialize_components()
        
        self.fitness_cache = {}
        self.cache_lock = threading.RLock()
        self.cache_hits = 0
        self.cache_misses = 0
        
        self.population = self.population_generator.create_initial_population_hybrid(
            self.population_size, self.heuristic_tax
        )

    def _initialize_components(self):
        self.cost_calculator = CostCalculator(
            self.duration_matrix, self.distance_matrix,
            self.max_trip_duration, self.max_trip_distance, self.time_to_stop,
            self.time_weight, self.distance_weight
        )
        
        self.parallel_evaluator = ParallelFitnessEvaluator(
            self.duration_matrix, self.distance_matrix, self.num_points,
            self.vehicle_max_points, self.max_trip_duration, self.max_trip_distance,
            self.time_to_stop, self.time_weight, self.distance_weight
        )
        
        self.population_generator = PopulationGenerator(
            self.duration_matrix, self.distance_matrix, self.points_coordinates,
            self.max_vehicles, self.vehicle_max_points, self.num_points
        )
        
        self.operators = VRPOperators(
            self.max_vehicles, self.vehicle_max_points, self.calculate_fitness
        )

    def calculate_fitness(self, solution: List[List[int]]) -> float:
        solution_str = str(solution).encode('utf-8')
        solution_key = hashlib.md5(solution_str).hexdigest()
        
        with self.cache_lock:
            if solution_key in self.fitness_cache:
                self.cache_hits += 1
                return self.fitness_cache[solution_key]
        
        self.cache_misses += 1
        fitness = self.cost_calculator.calculate_fitness(solution, self.max_vehicles, self.num_points)
        
        with self.cache_lock:
            if len(self.fitness_cache) < 50000:
                self.fitness_cache[solution_key] = fitness
            elif len(self.fitness_cache) >= 50000:
                self.fitness_cache.clear()
                self.fitness_cache[solution_key] = fitness
        
        return fitness

    def run(self, epoch_callback: Optional[Callable] = None) -> Tuple[List[List[int]], float]:
        logging.info("Starting VRP Genetic Algorithm execution")
        
        chunksize = max(1, self.population_size // (self.parallel_evaluator.cpu_count * 4))
        logging.info(f"Using optimized chunk size: {chunksize} for {self.parallel_evaluator.cpu_count} cores")

        logging.info("Evaluating initial population in parallel...")
        fitness_cache = self.parallel_evaluator.evaluate_population(self.population, chunksize)
        best_idx = min(range(len(fitness_cache)), key=lambda i: fitness_cache[i])
        best_solution = self.population[best_idx]
        best_cost = fitness_cache[best_idx]

        count_generations_without_improvement = 0
        count_generations_without_improvement_for_mutation = 0
        mutation_rate = self.mutation_rate

        for generation in range(self.generations):
            start_gen = time.time()
            new_population = []

            if count_generations_without_improvement >= self.max_no_improvement:
                logging.info(f"Generation {generation + 1}: Restarting population after {self.max_no_improvement} generations without improvement")
                self.population = self.population_generator.create_initial_population_hybrid(
                    self.population_size, self.heuristic_tax
                )
                fitness_cache = self.parallel_evaluator.evaluate_population(self.population, chunksize)
                count_generations_without_improvement = 0
                mutation_rate = self.mutation_rate

            if count_generations_without_improvement_for_mutation >= COUNT_GENERATIONS_WITHOUT_IMPROVEMENT_FOR_MUTATION:
                logging.info(f"Generation {generation + 1}: Increasing mutation rate")
                mutation_rate = min(MAX_MUTATION_RATE, mutation_rate * MUTATION_RATE_INCREASE_FACTOR)
                count_generations_without_improvement_for_mutation = 0

            self._apply_local_optimizations(generation, best_solution, best_cost)

            best_gen_idx = min(range(len(fitness_cache)), key=lambda i: fitness_cache[i])
            best_of_gen = self.population[best_gen_idx]
            new_population.append(best_of_gen)

            remaining_population = self.population_size - len(new_population)
            if remaining_population > 0:
                new_offspring = self._generate_offspring_parallel(remaining_population, mutation_rate)
                new_population.extend(new_offspring)

            new_fitness_cache = self.parallel_evaluator.evaluate_population(new_population, chunksize)

            self.population = new_population
            fitness_cache = new_fitness_cache

            current_best_idx = min(range(len(fitness_cache)), key=lambda i: fitness_cache[i])
            current_best = self.population[current_best_idx]
            current_best_cost = fitness_cache[current_best_idx]

            logging.info(f'Generation {generation + 1} - Best in population: {current_best_cost:.4f}')

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

            if epoch_callback:
                self._call_progress_callback(epoch_callback, generation + 1, best_solution)

        total_evaluations = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / total_evaluations * 100) if total_evaluations > 0 else 0
        
        logging.info(f"Algorithm completed. Final best cost: {best_cost:.4f}")
        logging.info(f"Fitness cache statistics: {self.cache_hits} hits, {self.cache_misses} misses ({cache_hit_rate:.1f}% hit rate)")
        logging.info(f"Total fitness evaluations reduced from {total_evaluations} to {self.cache_misses} ({total_evaluations - self.cache_misses} saved)")
        
        return best_solution, best_cost

    def _generate_offspring_parallel(self, num_offspring: int, mutation_rate: float) -> List[List[List[int]]]:        
        def generate_single_offspring(_):
            parents = self.operators.select_parents(self.population)
            child = self.operators.crossover(parents[0], parents[1])
            return self.operators.mutate(child, mutation_rate)
        
        max_workers = min(num_offspring, 20)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(generate_single_offspring, i) for i in range(num_offspring)]
            offspring = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        return offspring

    def _apply_local_optimizations(self, generation: int, best_solution: List[List[int]], 
                                 best_cost: float) -> Tuple[List[List[int]], float]:
        if generation % COUNT_GENERATIONS_WITHOUT_IMPROVEMENT_FOR_MUTATION == 0 and best_cost != float('inf'):
            logging.debug(f"Generation {generation + 1}: Applying inter-route local search")
            inter_opt = self.operators.inter_route_swap_search(
                best_solution, max_iter_without_improve=MAX_ITER_WITHOUT_IMPROVE_INTER_ROUTE
            )
            inter_cost = self.calculate_fitness(inter_opt)
            
            if inter_cost < best_cost:
                best_cost = inter_cost
                best_solution = inter_opt
                logging.info(f"Generation {generation + 1}: Inter-route search improved to {best_cost:.4f}")

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
                logging.info(f"Generation {generation + 1}: 2-opt improved to {best_cost:.4f}")

        return best_solution, best_cost

    def _call_progress_callback(self, callback: Callable, epoch: int, best_solution: List[List[int]]):
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

