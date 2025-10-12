import multiprocessing
from typing import List, Tuple, Set
from vrp.config import DEPOT_INDEX, TIME_DEPOT_STOP, DURATION_PENALTY_THRESHOLD, DISTANCE_PENALTY_THRESHOLD, PENALTY_MULTIPLIER


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
    DEPOT = DEPOT_INDEX
    
    if not solution:
        return float('inf')

    total_solution_cost = 0.0
    vehicles_used = len([r for r in solution if r])
    
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

        for i in range(len(depot_indices) - 1):
            start_idx = depot_indices[i]
            end_idx = depot_indices[i + 1]
            trip = vehicle_route[start_idx:end_idx + 1]
            
            if len(trip) <= 2:
                continue

            trip_cost, _, _ = _trip_cost_worker(trip)
            if trip_cost == float('inf'):
                return float('inf')

            total_solution_cost += trip_cost

            deliveries = [p for p in trip if p != DEPOT]
            visited_points.update(deliveries)
            all_delivery_points.extend(deliveries)

    if len(all_delivery_points) != len(visited_points):
        return float('inf')

    required = set(range(1, _worker_num_points))
    if visited_points != required:
        return float('inf')

    return total_solution_cost


def _trip_cost_worker(trip_points: List[int]) -> Tuple[float, int, int]:
    DEPOT = DEPOT_INDEX
    
    if len(trip_points) < 2 or trip_points[0] != DEPOT or trip_points[-1] != DEPOT:
        return float('inf'), 0, 0

    cur_duration = TIME_DEPOT_STOP
    cur_distance = 0
    last = trip_points[0]

    for p in trip_points[1:]:
        dtime = _worker_duration_matrix[last][p]
        ddist = _worker_distance_matrix[last][p]
        cur_duration += dtime
        cur_distance += ddist

        if p != DEPOT:
            cur_duration += _worker_time_to_stop
        elif p == DEPOT and last != DEPOT:
            cur_duration += TIME_DEPOT_STOP

        last = p

    if cur_duration > _worker_max_trip_duration or cur_distance > _worker_max_trip_distance:
        return float('inf'), cur_duration, cur_distance

    if len(trip_points) > _worker_vehicle_max_points + 2:
        return float('inf'), cur_duration, cur_distance

    duration_ratio = cur_duration / _worker_max_trip_duration
    distance_ratio = cur_distance / _worker_max_trip_distance
    
    penalty = 1.0
    if duration_ratio > DURATION_PENALTY_THRESHOLD or distance_ratio > DISTANCE_PENALTY_THRESHOLD:
        max_ratio = max(duration_ratio, distance_ratio)
        penalty += PENALTY_MULTIPLIER * ((max_ratio - DURATION_PENALTY_THRESHOLD) / (1.0 - DURATION_PENALTY_THRESHOLD)) ** 2

    trip_cost = penalty * (_worker_time_weight * duration_ratio + _worker_distance_weight * distance_ratio)
    return trip_cost, cur_duration, cur_distance


class CostCalculator:  
    def __init__(self, duration_matrix: List[List[float]], 
                 distance_matrix: List[List[float]],
                 max_trip_duration: int, 
                 max_trip_distance: int,
                 time_to_stop: int,
                 vehicle_max_points: int,
                 time_weight: float = 0.5, 
                 distance_weight: float = 0.5):
        self.duration_matrix = duration_matrix
        self.distance_matrix = distance_matrix
        self.max_trip_duration = max_trip_duration
        self.max_trip_distance = max_trip_distance
        self.time_to_stop = time_to_stop
        self.vehicle_max_points = vehicle_max_points
        self.time_weight = time_weight
        self.distance_weight = distance_weight
    
    def trip_cost_local(self, trip_points: List[int]) -> Tuple[float, int, int]:
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

        if len(trip_points) > self.vehicle_max_points + 2:
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
        total_solution_cost = 0.0
        visited_points = set()
        point_count = 0
        
        active_vehicles = 0
        for vehicle_route in solution:
            if vehicle_route:
                active_vehicles += 1
                
        if active_vehicles > max_vehicles:
            return float('inf')

        for vehicle_route in solution:
            if not vehicle_route:
                continue

            if vehicle_route[0] != DEPOT_INDEX or vehicle_route[-1] != DEPOT_INDEX:
                return float('inf')

            depot_indices = []
            for i, x in enumerate(vehicle_route):
                if x == DEPOT_INDEX:
                    depot_indices.append(i)
            
            if len(depot_indices) < 2:
                return float('inf')

            for i in range(len(depot_indices) - 1):
                start_idx = depot_indices[i]
                end_idx = depot_indices[i + 1]
                trip = vehicle_route[start_idx:end_idx + 1]
                
                if len(trip) <= 2:
                    continue
                    
                trip_cost = self._fast_trip_cost(trip)
                if trip_cost == float('inf'):
                    return float('inf')
                    
                total_solution_cost += trip_cost
                
                for p in trip[1:-1]:
                    if p in visited_points:
                        return float('inf')
                    visited_points.add(p)
                    point_count += 1

        if point_count != num_points - 1:
            return float('inf')

        return total_solution_cost

    def _fast_trip_cost(self, trip_points: List[int]) -> float:
        if len(trip_points) < 2:
            return float('inf')

        cur_duration = TIME_DEPOT_STOP
        cur_distance = 0
        last = trip_points[0]
        
        for p in trip_points[1:]:
            cur_duration += self.duration_matrix[last][p]
            cur_distance += self.distance_matrix[last][p]

            if p != DEPOT_INDEX:
                cur_duration += self.time_to_stop
            elif last != DEPOT_INDEX:
                cur_duration += TIME_DEPOT_STOP

            last = p

        if cur_duration > self.max_trip_duration or cur_distance > self.max_trip_distance:
            return float('inf')

        duration_ratio = cur_duration / self.max_trip_duration
        distance_ratio = cur_distance / self.max_trip_distance
        
        penalty = 1.0
        max_ratio = max(duration_ratio, distance_ratio)
        if max_ratio > DURATION_PENALTY_THRESHOLD:
            excess = (max_ratio - DURATION_PENALTY_THRESHOLD) / (1.0 - DURATION_PENALTY_THRESHOLD)
            penalty += PENALTY_MULTIPLIER * excess * excess

        return penalty * (self.time_weight * duration_ratio + self.distance_weight * distance_ratio)


class ParallelFitnessEvaluator:  
    def __init__(self, duration_matrix: List[List[float]], 
                 distance_matrix: List[List[float]],
                 num_points: int,
                 vehicle_max_points: int,
                 max_trip_duration: int, 
                 max_trip_distance: int,
                 time_to_stop: int,
                 time_weight: float = 0.5, 
                 distance_weight: float = 0.5):
        self.pool_init_args = (
            duration_matrix, distance_matrix, num_points,
            vehicle_max_points, max_trip_duration, max_trip_distance,
            time_to_stop, time_weight, distance_weight
        )
        self.cpu_count = max(1, multiprocessing.cpu_count())
    
    def evaluate_population(self, population: List[List[List[int]]], 
                          chunksize: int = None) -> List[float]:
        if chunksize is None:
            chunksize = len(population) // self.cpu_count if len(population) > self.cpu_count else 1
        
        with multiprocessing.Pool(processes=self.cpu_count, 
                                initializer=_init_worker, 
                                initargs=self.pool_init_args) as pool:
            fitness_scores = pool.map(fitness_worker, population, chunksize=chunksize)
        
        return fitness_scores