import multiprocessing
import random
import math
import logging
import time
from functools import partial
from copy import deepcopy

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# ---------- constantes ----------
POPULATION_HEURISTIC_TAX = 0.5
TWO_OPT_FREQUENCY = 10  # aplicar 2-opt no best solution a cada X gerações
TIME_DEPOT_STOP = 180  # 3 minutos em segundos por depósito/parada
COUNT_GENERATIONS_WITHOUT_IMPROVEMENT_FOR_MUTATION = 5
DEPOT_INDEX = 0

# ---------- variáveis globais do worker (inicializadas via initializer) ----------
_worker_duration_matrix = None
_worker_distance_matrix = None
_worker_num_points = None
_worker_vehicle_max_points = None
_worker_max_trip_duration = None
_worker_max_trip_distance = None
_worker_time_to_stop = None
_worker_time_weight = None
_worker_distance_weight = None


def _init_worker(duration_matrix, distance_matrix, num_points,
                 vehicle_max_points, max_trip_duration, max_trip_distance,
                 time_to_stop, time_weight, distance_weight):
    """Initializer para os workers do multiprocessing: carrega matrizes globalmente no worker."""
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


def fitness_worker(solution):
    """
    Versão enxuta da função de fitness para os workers: calcula o custo total de 'solution'
    usando as matrizes globais já carregadas no processo worker.
    Retorna float('inf') para soluções inviáveis.
    """
    DEPOT = DEPOT_INDEX
    # sanity checks
    if not solution:
        return float('inf')

    total_solution_cost = 0.0
    vehicles_used = len([r for r in solution if r])
    if vehicles_used > 0 and vehicles_used > 10000:
        # proteção contra valores absurdos (não deve ocorrer)
        return float('inf')

    visited_points = set()
    all_delivery_points = []

    for vehicle_route in solution:
        if not vehicle_route:
            continue

        if vehicle_route[0] != DEPOT or vehicle_route[-1] != DEPOT:
            return float('inf')

        depot_indices = [i for i, x in enumerate(vehicle_route) if x == DEPOT]
        if len(depot_indices) < 2:
            return float('inf')

        # para cada trip dentro da rota (pode haver vários DEPOT...DEPOT)
        for i in range(len(depot_indices) - 1):
            start_idx = depot_indices[i]
            end_idx = depot_indices[i + 1]
            trip = vehicle_route[start_idx:end_idx + 1]
            if len(trip) <= 2:
                # rota sem entregas nesse trip
                continue

            trip_cost, trip_duration, trip_distance = _trip_cost_worker(trip)
            if trip_cost == float('inf'):
                return float('inf')

            total_solution_cost += trip_cost

            deliveries = [p for p in trip if p != DEPOT]
            visited_points.update(deliveries)
            all_delivery_points.extend(deliveries)

    # duplicidade de pontos
    if len(all_delivery_points) != len(visited_points):
        return float('inf')

    required = set(range(1, _worker_num_points))
    if visited_points != required:
        return float('inf')

    return total_solution_cost


def _trip_cost_worker(trip_points):
    """
    Calcula custo (normalizado) de um trip [0, p1, ..., 0] usando matrizes globais no worker.
    Retorna (trip_cost, duration_seconds, distance_units).
    """
    DEPOT = DEPOT_INDEX
    if len(trip_points) < 2 or trip_points[0] != DEPOT or trip_points[-1] != DEPOT:
        return float('inf'), 0, 0

    cur_duration = TIME_DEPOT_STOP  # espera inicial no depósito
    cur_distance = 0
    last = trip_points[0]

    for p in trip_points[1:]:
        # busca nas matrizes globais (assumimos indexadas corretamente)
        dtime = _worker_duration_matrix[last][p]
        ddist = _worker_distance_matrix[last][p]
        cur_duration += dtime
        cur_distance += ddist

        if p != DEPOT:
            cur_duration += _worker_time_to_stop
        elif p == DEPOT and last != DEPOT:
            cur_duration += TIME_DEPOT_STOP

        last = p

    # validações de limite
    if cur_duration > _worker_max_trip_duration or cur_distance > _worker_max_trip_distance:
        return float('inf'), cur_duration, cur_distance

    duration_ratio = cur_duration / _worker_max_trip_duration
    distance_ratio = cur_distance / _worker_max_trip_distance
    penalty = 1.0
    if duration_ratio > 0.9 or distance_ratio > 0.9:
        penalty += 5 * ((max(duration_ratio, distance_ratio) - 0.9) / 0.1) ** 2

    trip_cost = penalty * (_worker_time_weight * duration_ratio + _worker_distance_weight * distance_ratio)
    return trip_cost, cur_duration, cur_distance


# ---------- CLASSE OTIMIZADA ----------
class VRPGeneticAlgorithm:
    def __init__(self, duration_matrix, distance_matrix, points: list,
                 max_vehicles: int, vehicle_max_points: int,
                 generations: int, population_size: int, population_heuristic_tax: float,
                 max_trip_duration: int, max_trip_distance: int, time_to_stop: int,
                 mutation_rate=0.05, max_no_improvement=50, time_weight=0.5, distance_weight=0.5):
        # parâmetros
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

        # dados
        self.duration_matrix = duration_matrix
        self.distance_matrix = distance_matrix
        self.points_coordinates = points
        self.num_points = len(points)
        self.time_weight = time_weight
        self.distance_weight = distance_weight

        # população inicial (híbrida heurística + aleatória)
        self.population = self.create_initial_population_hybrid()

        # prepare pool init args (para usar no run)
        self._pool_init_args = (self.duration_matrix, self.distance_matrix, self.num_points,
                                self.vehicle_max_points, self.max_trip_duration, self.max_trip_distance,
                                self.time_to_stop, self.time_weight, self.distance_weight)

    # ---------------- utilitários ----------------
    def _clean_route(self, route):
        """Remove depósitos duplicados (0, 0) e garante que rotas vazias retornem []."""
        if not route:
            return []
        # garante que começa e termina com DEPOT_INDEX (se possível)
        if route[0] != DEPOT_INDEX:
            # tenta ajustar
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

    def _trip_cost_local(self, trip_points):
        """
        Versão local de custo semelhante ao worker (para uso no processo principal).
        Retorna (trip_cost, duration, distance)
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
        if duration_ratio > 0.9 or distance_ratio > 0.9:
            penalty += 5 * ((max(duration_ratio, distance_ratio) - 0.9) / 0.1) ** 2

        trip_cost = penalty * (self.time_weight * duration_ratio + self.distance_weight * distance_ratio)
        return trip_cost, cur_duration, cur_distance

    # ---------------- geração de população ----------------
    def _create_initial_population_random_only(self):
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

    def create_initial_population_heuristic_optimized(self):
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
                # nearest neighbor
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

    def create_initial_population_hybrid(self):
        num_clustered = int(self.population_size * POPULATION_HEURISTIC_TAX)
        num_random = self.population_size - num_clustered

        population = []
        for _ in range(num_clustered):
            population.append(self.create_initial_population_heuristic_optimized())
        for _ in range(num_random):
            population.append(self._create_initial_population_random_only())

        return population

    # ---------------- fitness (local) ----------------
    def calculate_fitness(self, solution):
        """
        Cálculo completo de fitness (ponto de verdade no processo principal).
        Usado apenas quando necessário (por exemplo, validação / cálculo final).
        """
        total_solution_cost = 0.0
        vehicles_used = len([r for r in solution if r])
        if vehicles_used > self.max_vehicles:
            return float('inf')

        visited_points = set()
        all_delivery_points = []

        for vehicle_route in solution:
            if not vehicle_route:
                continue

            depot_indices = [i for i, x in enumerate(vehicle_route) if x == DEPOT_INDEX]
            if vehicle_route[0] != DEPOT_INDEX or vehicle_route[-1] != DEPOT_INDEX or len(depot_indices) < 2:
                return float('inf')

            for i in range(len(depot_indices) - 1):
                start_idx = depot_indices[i]
                end_idx = depot_indices[i + 1]
                trip = vehicle_route[start_idx:end_idx + 1]
                if len(trip) <= 2:
                    continue
                trip_cost, _, _ = self._trip_cost_local(trip)
                if trip_cost == float('inf'):
                    return float('inf')
                total_solution_cost += trip_cost
                deliveries = [p for p in trip if p != DEPOT_INDEX]
                visited_points.update(deliveries)
                all_delivery_points.extend(deliveries)

        if len(all_delivery_points) != len(visited_points):
            return float('inf')

        all_required_points = set(range(1, self.num_points))
        if visited_points != all_required_points:
            return float('inf')

        return total_solution_cost

    # ---------------- operadores genéticos ----------------
    def select_parents(self):
        tournament_size = min(5, len(self.population))
        parents = []
        for _ in range(2):
            competitors = random.sample(self.population, tournament_size)
            best = min(competitors, key=self._cached_fitness)
            parents.append(best)
        return parents

    def crossover(self, parent1, parent2):
        # mesma lógica do seu crossover original, com pequenas otimizações (shallow copies)
        if not parent1 or not parent2:
            return deepcopy(random.choice([parent1, parent2]))

        p1_points = [p for route in parent1 for p in route if p != DEPOT_INDEX]
        p2_points = [p for route in parent2 for p in route if p != DEPOT_INDEX]
        all_unique = list(set(p1_points + p2_points))
        random.shuffle(all_unique)
        points_to_assign = all_unique[:]

        child = []
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
            trips_stops = []
            for i in range(len(depot_indices) - 1):
                trip_points = parent_route[depot_indices[i]:depot_indices[i + 1] + 1]
                trips_stops.append(len([g for g in trip_points if g != DEPOT_INDEX]))
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
                # limpar duplicados
                final_route = [child_route[0]]
                for i in range(1, len(child_route)):
                    if child_route[i] == DEPOT_INDEX and child_route[i - 1] == DEPOT_INDEX:
                        continue
                    final_route.append(child_route[i])
                child.append(final_route)
            else:
                child.append([])

        # distribuir remanescentes
        temp_points = points_to_assign[:]
        points_removed = []
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
                child[route_idx] = self._clean_route(child[route_idx])

        points_to_assign = [p for p in points_to_assign if p not in points_removed]

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
                        child[self.max_vehicles - 1] = self._clean_route(child[self.max_vehicles - 1])

        child = child[:self.max_vehicles]
        while len(child) < self.max_vehicles:
            child.append([])

        return child

    # ---------------- delta-cost helpers ----------------
    def _find_trip_bounds(self, route, pos):
        """
        Dado um índice pos dentro de 'route', retorna (start_idx, end_idx) do trip que contém pos.
        Assumimos que route tem DEPOT markers.
        """
        # encontra depósito anterior (inclusive) e próximo depósito (inclusive)
        start = pos
        while start > 0 and route[start] != DEPOT_INDEX:
            start -= 1
        # agora start aponta para depot (ou 0)
        end = pos
        n = len(route)
        while end < n - 1 and route[end] != DEPOT_INDEX:
            end += 1
        # garante que end é índice do depósito final do trip
        if route[end] != DEPOT_INDEX:
            # se não encontrou, assume final
            end = n - 1
        return start, end

    def _trip_sublist(self, route, start_idx, end_idx):
        return route[start_idx:end_idx + 1]

    # calc delta ao mover um único ponto de source_route[pos_idx] para target_route insertion at insert_idx
    def _delta_move_one_point(self, solution, s_idx, pos_idx, t_idx, insert_idx):
        """
        Calcula a variação de custo (delta) ao mover o ponto em solution[s_idx][pos_idx]
        para solution[t_idx] na posição insert_idx (index na lista do route após inserção).
        Retorna (delta, feasible_bool).
        NOTE: pos_idx e insert_idx são índices *na rota atual* (antes de mutação).
        """
        DEPOT = DEPOT_INDEX

        source_route = solution[s_idx]
        target_route = solution[t_idx]

        # índice inválido
        if pos_idx < 0 or pos_idx >= len(source_route):
            return 0, False
        point = source_route[pos_idx]
        if point == DEPOT:
            return 0, False

        # encontra trip bounds na rota de origem
        s_trip_start, s_trip_end = self._find_trip_bounds(source_route, pos_idx)
        s_trip = self._trip_sublist(source_route, s_trip_start, s_trip_end)

        # cria s_trip_after (removendo o ponto)
        s_trip_after = [p for i, p in enumerate(s_trip) if not (s_trip_start + i == pos_idx)]
        s_trip_after = self._clean_route(s_trip_after)

        # calcula custo antigo e novo do trip origem
        old_src_cost, old_src_dur, old_src_dist = self._trip_cost_local(s_trip)
        if old_src_cost == float('inf'):
            return 0, False
        new_src_cost, new_src_dur, new_src_dist = (0, 0, 0)
        if s_trip_after:
            new_src_cost, new_src_dur, new_src_dist = self._trip_cost_local(s_trip_after)
            if new_src_cost == float('inf'):
                return 0, False
        else:
            new_src_cost = 0.0

        # agora trata destino
        if not target_route:
            # inserir em rota vazia cria trip [0, point, 0]
            t_trip_old = []
            t_trip_new = [DEPOT, point, DEPOT]
        else:
            # insert_idx é índice de inserção na lista target_route (p.ex. 1..len-1)
            # precisamos localizar o trip dentro de target_route onde insert_idx vai cair
            if insert_idx < 0 or insert_idx > len(target_route):
                return 0, False
            # se insert está no final (antes do depot final), ajusta
            # converte insert idx para índice dentro do trip
            # descobrimos o trip bounds que contém insert_idx (se insert_idx for depot, usamos trip anterior)
            t_insert_index = insert_idx
            if t_insert_index == len(target_route):
                # inserir no final -> no último trip (antes do último depot)
                t_insert_index = len(target_route) - 1

            t_trip_start, t_trip_end = self._find_trip_bounds(target_route, max(0, t_insert_index - 1))
            t_trip = self._trip_sublist(target_route, t_trip_start, t_trip_end)
            # cria nova trip com o ponto inserido na posição relativa
            rel_pos = insert_idx - t_trip_start
            if rel_pos < 1:
                rel_pos = 1
            if rel_pos > len(t_trip) - 1:
                rel_pos = len(t_trip) - 1
            t_trip_new = t_trip[:rel_pos] + [point] + t_trip[rel_pos:]
            t_trip_old = t_trip

        # calcula custos antigo e novo do trip destino
        old_tgt_cost = 0.0
        if target_route:
            if t_trip_old and len(t_trip_old) > 1:
                old_tgt_cost, _, _ = self._trip_cost_local(t_trip_old)
                if old_tgt_cost == float('inf'):
                    return 0, False
        else:
            old_tgt_cost = 0.0

        new_tgt_cost, _, _ = self._trip_cost_local(t_trip_new)
        if new_tgt_cost == float('inf'):
            return 0, False

        delta = (new_src_cost + new_tgt_cost) - (old_src_cost + old_tgt_cost)
        return delta, True

    # ---------------- busca local otimizada (inter-route delta-cost) ----------------
    def inter_route_swap_search(self, solution, max_iter_without_improve=1):
        """
        Busca local inter-route com delta-cost.
        Retorna solução melhorada (ou a mesma se não encontrar melhora).
        max_iter_without_improve controla quantas melhorias consecutivas buscar (default 1 com early stop).
        """
        current_solution = [r[:] for r in solution]
        current_cost = self.calculate_fitness(current_solution)
        if current_cost == float('inf'):
            # tentativa fraca de consertar: retorna original
            return current_solution

        improved_any = False
        iter_no_improve = 0

        # Pré-cálculo índices de pontos (não-depot) por rota para iterar
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
                    # para cada rota alvo
                    for t_idx, t_route in enumerate(current_solution):
                        # pre-calc inserções possíveis:
                        if s_idx == t_idx:
                            # ao mover dentro da mesma rota, apenas tentar trocar posições (pouco ganho aqui) -> pular
                            continue
                        # define possíveis posições de inserção na rota alvo:
                        if not t_route:
                            insert_positions = [1]  # cria [0, x, 0]
                        else:
                            # inserir entre índices [1 .. len-1]
                            insert_positions = range(1, len(t_route))
                        # testa algumas inserções (podemos limitar para desempenho — testaremos todas por hora)
                        for ins in insert_positions:
                            delta, feasible = self._delta_move_one_point(current_solution, s_idx, pos, t_idx, ins)
                            if not feasible:
                                continue
                            if delta < best_delta:
                                best_delta = delta
                                best_move = (s_idx, pos, t_idx, ins, delta)
                                # early break se encontrar melhoria significativa
                                # (usar a primeira melhora encontrada para velocidade)
                                # Aqui optamos por capturar a melhor dentro do scanning atual
                        # fim insert_positions
                    # fim target routes
                # fim pos in source
            # fim s_idx

            if best_move and best_move[4] < -1e-9:
                s_idx, pos, t_idx, ins, d = best_move
                # aplica movimento criando cópias rasas das rotas afetadas
                s_route = current_solution[s_idx][:]
                point_to_move = s_route[pos]
                s_route.pop(pos)
                s_route = self._clean_route(s_route)

                t_route = current_solution[t_idx][:]
                if not t_route:
                    t_route = [DEPOT_INDEX, point_to_move, DEPOT_INDEX]
                else:
                    # ajusta posição ins (pode exceder após remoção de outro)
                    if ins > len(t_route):
                        ins = len(t_route)
                    t_route.insert(ins, point_to_move)
                    t_route = self._clean_route(t_route)

                # atualiza solução
                current_solution[s_idx] = s_route
                current_solution[t_idx] = t_route

                current_cost += d
                improved_any = True
                iter_no_improve = 0
                # continue procurando novas melhorias (ou early stop se preferir)
            else:
                iter_no_improve += 1
                break

        return current_solution

    # ---------------- mutação ----------------
    def mutate(self, solution, mutation_rate):
        is_mutated = random.random() < mutation_rate
        if not is_mutated:
            return solution

        # 50% 2-opt global (restringido a pontos)
        if random.random() < 0.5:
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
                new_solution.append(self._clean_route(reconstructed))
            while len(new_solution) < self.max_vehicles:
                new_solution.append([])
            # aceita se melhorar
            if self.calculate_fitness(new_solution) < self.calculate_fitness(solution):
                return new_solution
            return solution
        else:
            # Realocação inteligente (inter-route delta cost)
            return self.inter_route_swap_search(solution)

    # ---------------- 2-opt intra-route (aplicar seletivamente) ----------------
    def two_opt_local_search(self, solution, max_improvements=1):
        """
        Busca 2-opt aplicada apenas à solução fornecida (geralmente best_solution).
        Limitamos tentativas para rotas grandes e aplicamos early stop.
        """
        improved_solution = [r[:] for r in solution]
        total_cost = self.calculate_fitness(improved_solution)
        if total_cost == float('inf'):
            return improved_solution

        improvements = 0
        # itera pelas rotas
        for route_idx in range(len(improved_solution)):
            route = improved_solution[route_idx]
            points_only = [g for g in route if g != DEPOT_INDEX]
            if len(points_only) < 5:
                continue  # rotas pequenas não ganham muito com 2-opt

            # tenta combinações i, j
            changed = False
            for i in range(len(points_only) - 1):
                for j in range(i + 1, len(points_only)):
                    temp_points = points_only[:]
                    temp_points[i:j + 1] = temp_points[i:j + 1][::-1]
                    # reconstruir route mantendo múltiplos trips (preservar estruturas internas)
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
                    final_route = self._clean_route(temp_route)
                    temp_solution = [r[:] for r in improved_solution]
                    temp_solution[route_idx] = final_route
                    new_cost = self.calculate_fitness(temp_solution)
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

    # ---------------- run (loop principal) ----------------
    def _cached_fitness(self, sol):
        """
        wrapper para calcular fitness sem usar Pool (cache não global).
        será substituído dinamicamente por um binding no run() para usar cache local.
        """
        return self.calculate_fitness(sol)

    def run(self, epoch_callback: callable = None):
        """Executa o loop do algoritmo genético otimizado."""
        cpu_count = max(1, multiprocessing.cpu_count())
        chunksize = self.population_size // cpu_count if self.population_size > cpu_count else 1

        # fitness cache local por indivíduo (índice) para evitar recalcular repetidamente
        fitness_cache = [None] * len(self.population)
        for i, sol in enumerate(self.population):
            fitness_cache[i] = self.calculate_fitness(sol)

        best_idx = min(range(len(fitness_cache)), key=lambda i: fitness_cache[i])
        best_solution = self.population[best_idx]
        best_cost = fitness_cache[best_idx]

        count_generations_without_improvement = 0
        count_generations_without_improvement_for_mutation = 0
        mutation_rate = self.mutation_rate

        # cria pool com initializer para workers
        pool = multiprocessing.Pool(processes=cpu_count, initializer=_init_worker, initargs=self._pool_init_args)

        # binder para usar o fitness_worker via pool.map (funciona como referência)
        try:
            for generation in range(self.generations):
                start_gen = time.time()
                new_population = []

                # reinicializa população se estagnar
                if count_generations_without_improvement >= self.max_no_improvement:
                    logging.info("Geração %d: Sem melhoria por %d gerações. Reiniciando população.", generation + 1, self.max_no_improvement)
                    self.population = self.create_initial_population_hybrid()
                    fitness_cache = [self.calculate_fitness(sol) for sol in self.population]
                    count_generations_without_improvement = 0
                    mutation_rate = self.mutation_rate

                # aumento adaptativo de mutação
                if count_generations_without_improvement_for_mutation >= COUNT_GENERATIONS_WITHOUT_IMPROVEMENT_FOR_MUTATION:
                    logging.info("Geração %d: Aumentando taxa de mutação.", generation + 1)
                    mutation_rate = min(0.5, mutation_rate * 1.1)
                    count_generations_without_improvement_for_mutation = 0

                # Inter-route local search no best_solution (rápida) usando delta-cost
                if generation % COUNT_GENERATIONS_WITHOUT_IMPROVEMENT_FOR_MUTATION == 0 and best_cost != float('inf'):
                    logging.debug("Geração %d: Inter-route local search no best_solution.", generation + 1)
                    inter_opt = self.inter_route_swap_search(best_solution, max_iter_without_improve=1)
                    inter_cost = self.calculate_fitness(inter_opt)
                    if inter_cost < best_cost:
                        best_cost = inter_cost
                        best_solution = inter_opt
                        count_generations_without_improvement = 0
                        count_generations_without_improvement_for_mutation = 0
                        logging.info("Geração %d: Inter-swap melhorou best_solution para %.4f", generation + 1, best_cost)

                # 2-opt aplicada somente ao best_solution a cada TWO_OPT_FREQUENCY
                if generation % TWO_OPT_FREQUENCY == 0 and best_cost != float('inf'):
                    # aplica apenas se existir rota grande
                    if any(len([p for p in r if p != DEPOT_INDEX]) >= 6 for r in best_solution):
                        logging.debug("Geração %d: Aplicando 2-opt no best_solution.", generation + 1)
                        local_opt = self.two_opt_local_search(best_solution, max_improvements=1)
                        local_cost = self.calculate_fitness(local_opt)
                        if local_cost < best_cost:
                            best_cost = local_cost
                            best_solution = local_opt
                            count_generations_without_improvement = 0
                            logging.info("Geração %d: 2-opt melhorou best_solution para %.4f", generation + 1, best_cost)

                # seleção e geração - elitismo simples (mantém melhor da geração)
                best_gen_idx = min(range(len(fitness_cache)), key=lambda i: fitness_cache[i])
                best_of_gen = self.population[best_gen_idx]
                new_population.append(best_of_gen)

                while len(new_population) < self.population_size:
                    parent1, parent2 = self.select_parents()
                    child = self.crossover(parent1, parent2)
                    mutated = self.mutate(child, mutation_rate)
                    new_population.append(mutated)

                # calcula fitness da nova população com pool (workers)
                # usa fitness_worker via pool.map (matrizes já no worker)
                new_fitness_cache = pool.map(fitness_worker, new_population, chunksize=chunksize)

                # atualiza população e cache
                self.population = new_population
                fitness_cache = new_fitness_cache

                current_best_idx = min(range(len(fitness_cache)), key=lambda i: fitness_cache[i])
                current_best = self.population[current_best_idx]
                current_best_cost = fitness_cache[current_best_idx]
                logging.info('Geração %d - Custo da melhor solução na população: %.4f', generation + 1, current_best_cost)

                # atualiza melhor global
                if current_best_cost < best_cost:
                    best_solution = current_best
                    best_cost = current_best_cost
                    count_generations_without_improvement = 0
                    count_generations_without_improvement_for_mutation = 0
                else:
                    count_generations_without_improvement += 1
                    count_generations_without_improvement_for_mutation += 1

                logging.info('Geração %d - Melhor Custo Global: %.4f (tempo ger: %.2fs)', generation + 1, best_cost, time.time() - start_gen)

                # callback de progresso
                if epoch_callback:
                    vehicle_points = []
                    for i, route in enumerate(best_solution):
                        if route:
                            trip_cost, trip_duration, trip_distance = self._trip_cost_local(route)
                            vehicle_points.append({
                                'vehicle_id': i + 1,
                                'points': len([p for p in route if p != DEPOT_INDEX]),
                                'route': route,
                                'distance': round(trip_distance / 1000, 2),
                                'duration': round(trip_duration / 60, 1),
                                'cost': round(trip_cost, 6) if trip_cost != float('inf') else None
                            })
                    epoch_callback(epoch=generation + 1, vehicles=len(vehicle_points), vehicle_data=vehicle_points)

            # fim for generations
        finally:
            pool.close()
            pool.join()

        return best_solution, best_cost


# ---------------- exemplo de uso ----------------
if __name__ == "__main__":
    # exemplo sintético rápido para testar (ajuste conforme necessário)
    random.seed(0)

    # parâmetros
    N_POINTS = 30  # comece com 30; escale para 60/100 conforme tuning
    MAX_VEHICLES = 6
    VEHICLE_MAX_POINTS = 8
    GENERATIONS = 50
    POPULATION_SIZE = 40

    # gera matrizes sintéticas (tempo em segundos e distância arbitrária)
    # matriz dimensão (N_POINTS) incluindo depot no índice 0
    def gen_matrix(n):
        m = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    m[i][j] = 0
                else:
                    # tempo/distance aleatórios razoáveis
                    val = random.randint(60, 600)  # entre 1 e 10 minutos
                    m[i][j] = val
        return m

    # gera coordenadas fictícias (apenas para heurística de varredura)
    points = [(0.0, 0.0)]
    for i in range(1, N_POINTS):
        points.append((random.random() * 0.1, random.random() * 0.1))

    duration_mtx = gen_matrix(N_POINTS)
    distance_mtx = gen_matrix(N_POINTS)

    # parâmetros de limite da viagem
    MAX_TRIP_DURATION = 4 * 3600  # 4 horas
    MAX_TRIP_DISTANCE = 400000  # unidades arbitrárias

    ga = VRPGeneticAlgorithm(duration_matrix=duration_mtx,
                             distance_matrix=distance_mtx,
                             points=points,
                             max_vehicles=MAX_VEHICLES,
                             vehicle_max_points=VEHICLE_MAX_POINTS,
                             generations=GENERATIONS,
                             population_size=POPULATION_SIZE,
                             population_heuristic_tax=POPULATION_HEURISTIC_TAX,
                             max_trip_duration=MAX_TRIP_DURATION,
                             max_trip_distance=MAX_TRIP_DISTANCE,
                             time_to_stop=120,  # 2 minutos por parada
                             mutation_rate=0.07,
                             max_no_improvement=30,
                             time_weight=0.6, distance_weight=0.4)

    def progress_cb(epoch, vehicles, vehicle_data):
        logging.info("Epoch %d - Vehicles: %d - Exemplo vehicle_data[0]: %s", epoch, vehicles, vehicle_data[0] if vehicle_data else {})

    start = time.time()
    best_sol, best_cost = ga.run(epoch_callback=progress_cb)
    end = time.time()

    logging.info("Melhor custo final: %.4f", best_cost)
    logging.info("Melhor solução (rotas):")
    for idx, r in enumerate(best_sol):
        if r:
            logging.info("Veículo %d: pontos=%d route=%s", idx + 1, len([p for p in r if p != DEPOT_INDEX]), r)
    logging.info("Tempo total: %.2fs", end - start)
