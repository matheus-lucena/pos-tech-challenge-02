import multiprocessing
import random
import math
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

POPULATION_HEURISTIC_TAX = 0.5
TWO_OPT_FREQUENCY = 10 # Apply local search every X generations
TIME_DEPOT_STOP = 180 # 3 minutes in seconds per stop

COUNT_GENERATIONS_WITHOUT_IMPROVEMENT_FOR_MUTATION = 5

DEPOT_INDEX = 0

# --- CLASSE DO ALGORITMO GENÉTICO VRP ---
class VRPGeneticAlgorithm:
    def __init__(self, duration_matrix, distance_matrix, points: list, max_vehicles: int, vehicle_max_points: int, generations: int, population_size: int, population_heuristic_tax: float, max_trip_duration: int, max_trip_distance: int, time_to_stop: int, mutation_rate=0.05, max_no_improvement=50, time_weight=0.5, distance_weight=0.5):
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
        self.population = self.create_initial_population_hybrid() 

    # --- FUNÇÃO AUXILIAR DE LIMPEZA DE ROTA ---
    def _clean_route(self, route):
        """Remove depósitos duplicados (0, 0) e garante que rotas vazias retornem []."""
        if not route:
            return []
            
        # Garante que começa e termina com DEPOT_INDEX
        if route[0] != DEPOT_INDEX or route[-1] != DEPOT_INDEX:
            # Rota malformada, mas tentamos limpar duplicados internos primeiro
            pass 
        
        clean_route = [route[0]]
        for p in route[1:]:
            # Evita duplicidade e garante que apenas o depot final é o último
            if p == DEPOT_INDEX and clean_route[-1] == DEPOT_INDEX:
                continue
            clean_route.append(p)
            
        # Se a rota é [0, 0] ou apenas [0], retorna vazia.
        if len(clean_route) <= 2 and all(p == DEPOT_INDEX for p in clean_route):
            return []
            
        return clean_route

    # --- MÉTODOS DE GERAÇÃO DE POPULAÇÃO ---

    def _create_initial_population_random_only(self):
        """Método original (Round Robin) para gerar a parte aleatória da população."""
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
        """Combina Varredura para agrupar e Vizinho Mais Próximo para sequenciar, respeitando MAX_VEHICLES."""
        
        depot_lat, depot_lon = self.points_coordinates[DEPOT_INDEX]
        all_points = list(range(1, self.num_points))
        
        # 1. Agrupamento por Ângulo (Sweep)
        points_angles = []
        for i in all_points:
            lat, lon = self.points_coordinates[i]
            angle = math.atan2(lat - depot_lat, lon - depot_lon)
            points_angles.append((angle, i))

        points_angles.sort()
        sorted_points = [point_idx for _, point_idx in points_angles]
        
        # 2. Criação de clusters/rotas respeitando MAX_VEHICLES
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
                
                # Sequenciamento por Vizinho Mais Próximo (Nearest Neighbor)
                while remaining_points:
                    best_next_point = -1
                    min_distance = float('inf')
                    
                    for next_point in remaining_points:
                        dist = self.distance_matrix[current_location][next_point]
                        if dist < min_distance:
                            min_distance = dist
                            best_next_point = next_point
                    
                    if best_next_point != -1:
                        current_trip.append(best_next_point)
                        remaining_points.remove(best_next_point)
                        current_location = best_next_point
                    else:
                        break 
                        
                if len(current_trip) > 1:
                    current_trip.append(DEPOT_INDEX)
                    vehicle_route.extend(current_trip[1:])
            
            if len(vehicle_route) > 2:
                final_solution.append(vehicle_route)

        # 3. Finalização: Garante exatamente MAX_VEHICLES rotas
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
            solution = self.create_initial_population_heuristic_optimized()
            population.append(solution)
            
        for _ in range(num_random):
            solution = self._create_initial_population_random_only()
            population.append(solution)
            
        return population

    # --- FUNÇÕES DE CUSTO ---

    def _get_trip_cost(self, trip_points):
        """
        Calcula o custo normalizado e a viabilidade de uma única viagem [0, p1, ..., 0].
        O custo agora penaliza fortemente viagens próximas do limite e favorece viagens mais balanceadas.
        """
        if len(trip_points) < 2 or trip_points[0] != DEPOT_INDEX or trip_points[-1] != DEPOT_INDEX:
            return float('inf'), 0, 0

        current_trip_duration = 0
        current_trip_distance = 0
        current_trip_stops = len([p for p in trip_points if p != DEPOT_INDEX])
        
        if current_trip_stops > self.vehicle_max_points:
            return float('inf'), 0, 0

        depot_indices = [i for i, x in enumerate(trip_points) if x == DEPOT_INDEX]
        if len(depot_indices) < 2:
            return float('inf'), 0, 0
        
        current_trip_duration += TIME_DEPOT_STOP

        for i in range(len(depot_indices) - 1):
            start_idx = depot_indices[i]
            end_idx = depot_indices[i+1]
            route_segment = trip_points[start_idx:end_idx+1]
            
            last_point_idx = route_segment[0] 
            
            for point_idx in route_segment[1:]:
                current_trip_duration += self.duration_matrix[last_point_idx][point_idx]
                current_trip_distance += self.distance_matrix[last_point_idx][point_idx]
                
                if point_idx != DEPOT_INDEX:
                    current_trip_duration += self.time_to_stop
                elif point_idx == DEPOT_INDEX and last_point_idx != DEPOT_INDEX:
                    current_trip_duration += TIME_DEPOT_STOP
                    
                last_point_idx = point_idx

        # Penalidade suave para viagens próximas do limite (exponencial)
        duration_ratio = current_trip_duration / self.max_trip_duration
        distance_ratio = current_trip_distance / self.max_trip_distance

        if current_trip_duration > self.max_trip_duration or current_trip_distance > self.max_trip_distance:
            return float('inf'), 0, 0

        # Penalidade extra para viagens que chegam muito perto do limite (exponencial)
        penalty = 1.0
        if duration_ratio > 0.9 or distance_ratio > 0.9:
            penalty += 5 * ((max(duration_ratio, distance_ratio) - 0.9) / 0.1) ** 2

        # Custo ponderado e penalizado
        trip_cost = penalty * (
            self.time_weight * duration_ratio +
            self.distance_weight * distance_ratio
        )

        return trip_cost, current_trip_duration, current_trip_distance

    def calculate_fitness(self, solution):
        """
        Calcula a aptidão (custo) da solução total. Inclui as Restrições Rígidas de Frota e Cobertura.
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
                end_idx = depot_indices[i+1]
                trip_points = vehicle_route[start_idx:end_idx+1]
                
                if len(trip_points) <= 2:
                    continue

                trip_cost, _, _ = self._get_trip_cost(trip_points)
                
                if trip_cost == float('inf'):
                    return float('inf')

                total_solution_cost += trip_cost
                
                delivery_points = [p for p in trip_points if p != DEPOT_INDEX]
                visited_points.update(delivery_points)
                all_delivery_points.extend(delivery_points)
        
        # --- RESTRIÇÃO RÍGIDA 4: DUPLICIDADE DE PONTOS ---
        if len(all_delivery_points) != len(visited_points):
            return float('inf')
            
        # --- RESTRIÇÃO RÍGIDA 5: COBERTURA COMPLETA ---
        all_required_points = set(range(1, self.num_points))
        if visited_points != all_required_points:
            return float('inf')

        return total_solution_cost


    # --- OPERADORES GENÉTICOS ---

    def select_parents(self):
        """Seleção por torneio para escolher os pais."""
        tournament_size = 5
        parents = []
        for _ in range(2):
            tournament_competitors = random.sample(self.population, tournament_size)
            best_competitor = min(tournament_competitors, key=self.calculate_fitness)
            parents.append(best_competitor)
        return parents

    def crossover(self, parent1, parent2):
        """
        Crossover VRP. Garante a unicidade e o respeito à capacidade de pontos por viagem.
        """
        if not parent1 or not parent2:
            return [r[:] for r in random.choice([parent1, parent2])]

        parent1_points = [p for route in parent1 for p in route if p != DEPOT_INDEX]
        parent2_points = [p for route in parent2 for p in route if p != DEPOT_INDEX]
        all_unique_points = list(set(parent1_points + parent2_points))
        random.shuffle(all_unique_points)
        
        child = []
        points_to_assign = all_unique_points[:]
        
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
                trip_points = parent_route[depot_indices[i]:depot_indices[i+1]+1]
                trips_stops.append(len(trip_points) - 2)

            child_route = []
            
            for num_stops in trips_stops:
                current_trip_points = []
                
                for _ in range(num_stops):
                    if points_to_assign:
                        current_trip_points.append(points_to_assign.pop(0))
                    else:
                        break
                
                if current_trip_points:
                    current_trip = [DEPOT_INDEX] + current_trip_points + [DEPOT_INDEX]
                    child_route.extend(current_trip)
            
            if child_route:
                # Limpa DEPOT_INDEX duplicados
                final_route = [child_route[0]]
                for i in range(1, len(child_route)):
                    if child_route[i] == DEPOT_INDEX and child_route[i-1] == DEPOT_INDEX:
                        continue
                    final_route.append(child_route[i])

                child.append(final_route)
            else:
                 child.append([])


        # 4. Distribuição de pontos remanescentes (GARANTIA DE COBERTURA TOTAL)
        points_removed = []
        
        # 4a. Tenta preencher rotas existentes (criando novas viagens)
        temp_points = points_to_assign[:]
        
        for route_idx, route in enumerate(child):
            if not temp_points: break
            
            if not route: continue 
                
            while temp_points:
                new_trip_points = temp_points[:self.vehicle_max_points] 
                
                if not new_trip_points: break
                    
                temp_points = temp_points[self.vehicle_max_points:]
                
                new_trip = [DEPOT_INDEX] + new_trip_points + [DEPOT_INDEX]
                child[route_idx].extend(new_trip)
                points_removed.extend(new_trip_points)
                
                # Limpa depósitos duplicados
                child[route_idx] = self._clean_route(child[route_idx])

        # Atualiza a lista principal de pontos restantes
        points_to_assign = [p for p in points_to_assign if p not in points_removed]


        # 4b. Cria novas rotas/veículos para pontos restantes (GARANTINDO COBERTURA)
        while points_to_assign:
            
            target_idx = len(child)
            
            if target_idx < self.max_vehicles:
                # Cria um novo veículo
                new_route = [DEPOT_INDEX]
                current_stops = 0
                
                while current_stops < self.vehicle_max_points and points_to_assign:
                    new_route.append(points_to_assign.pop(0))
                    current_stops += 1
                
                if len(new_route) > 1:
                    new_route.append(DEPOT_INDEX)
                    child.append(new_route)
            
            else:
                # MAX_VEHICLES atingido: insere os pontos restantes no último veículo
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
                         # Limpa depósitos duplicados
                         child[self.max_vehicles - 1] = self._clean_route(child[self.max_vehicles - 1])


        # 5. Finalização e garantia de tamanho do cromossomo
        child = child[:self.max_vehicles]
        while len(child) < self.max_vehicles:
            child.append([])

        return child

    def inter_route_swap_search(self, solution):
        """
        Busca local Inter-Route (One-Point Move) otimizada para desempenho máximo.
        Utiliza early stopping, minimiza cópias e evita recomputações desnecessárias.
        """
        current_solution = [r[:] for r in solution]
        current_cost = self.calculate_fitness(current_solution)
        improved = True

        # Pré-calcula índices de pontos não-depósito para cada rota
        def get_point_indices(route):
            return [i for i, p in enumerate(route) if p != DEPOT_INDEX]

        while improved:
            improved = False
            best_move = None
            min_new_cost = current_cost

            # Para cada rota de origem
            for source_route_idx, source_route in enumerate(current_solution):
                if not source_route or len(source_route) < 3:
                    continue
                source_point_indices = get_point_indices(source_route)

                # Para cada ponto na rota de origem
                for point_idx_in_route in source_point_indices:
                    point_to_move = source_route[point_idx_in_route]

                    # Para cada rota alvo
                    for target_route_idx, target_route in enumerate(current_solution):
                        # Define posições de inserção
                        if source_route_idx == target_route_idx:
                            insert_positions = [i for i in range(1, len(target_route))
                                                if i != point_idx_in_route and i != point_idx_in_route + 1]
                        elif not target_route:
                            insert_positions = [1]
                        else:
                            insert_positions = range(1, len(target_route))

                        for insert_index in insert_positions:
                            # Evita cópias desnecessárias: só copia se for testar
                            temp_solution = current_solution
                            # Remove ponto da origem
                            temp_source = temp_solution[source_route_idx][:]
                            temp_source.pop(point_idx_in_route)
                            temp_source = self._clean_route(temp_source)
                            # Insere ponto na rota alvo
                            temp_target = temp_solution[target_route_idx][:]
                            if not temp_target:
                                if len([r for r in temp_solution if r]) < self.max_vehicles or target_route_idx == source_route_idx:
                                    temp_target = [DEPOT_INDEX, point_to_move, DEPOT_INDEX]
                                else:
                                    continue
                            else:
                                temp_target.insert(insert_index, point_to_move)
                                temp_target = self._clean_route(temp_target)
                            # Monta nova solução
                            temp_new_solution = temp_solution[:]
                            temp_new_solution[source_route_idx] = temp_source
                            temp_new_solution[target_route_idx] = temp_target

                            # Calcula custo
                            new_cost = self.calculate_fitness(temp_new_solution)
                            if new_cost < min_new_cost:
                                min_new_cost = new_cost
                                best_move = (source_route_idx, temp_source, target_route_idx, temp_target)
                                improved = True
                                # Early stopping: aplica primeira melhoria encontrada
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

            # Aplica melhor movimento encontrado
            if improved and best_move:
                s_idx, s_route, t_idx, t_route = best_move
                current_solution = current_solution[:]
                current_solution[s_idx] = s_route
                current_solution[t_idx] = t_route
                current_cost = min_new_cost

        return current_solution

    def mutate(self, solution, mutation_rate):
        """
        Operador de mutação híbrido: 50% Inversão (2-Opt) / 50% Realocação Inteligente (Best Insertion).
        A Realocação Inteligente é feita pela busca local inter-rotas.
        """
        is_mutated = random.random() < mutation_rate
        if not is_mutated:
            return solution

        # 50% de chance de 2-Opt (Intra-Route)
        if random.random() < 0.5:
            # Lógica de 2-Opt Global (restringida a pontos)
            points_only = [gene for route in solution for gene in route if gene != DEPOT_INDEX]
            if len(points_only) < 2: return solution
            
            idx1, idx2 = sorted(random.sample(range(len(points_only)), 2))
            points_only[idx1:idx2] = points_only[idx1:idx2][::-1]

            new_solution = []
            points_iter = iter(points_only)
            
            for route in solution:
                if not route:
                    new_solution.append([])
                    continue
                
                reconstructed_route = []
                depot_indices = [i for i, x in enumerate(route) if x == DEPOT_INDEX]
                
                for i in range(len(depot_indices) - 1):
                    start_idx = depot_indices[i]
                    end_idx = depot_indices[i+1]
                    trip_points = route[start_idx:end_idx+1]
                    num_delivery_points = len([gene for gene in trip_points if gene != DEPOT_INDEX])
                    
                    current_trip = [DEPOT_INDEX]
                    try:
                        for _ in range(num_delivery_points):
                            current_trip.append(next(points_iter))
                        current_trip.append(DEPOT_INDEX)
                        reconstructed_route.extend(current_trip)
                    except StopIteration:
                        break

                new_solution.append(self._clean_route(reconstructed_route))

            while len(new_solution) < self.max_vehicles:
                new_solution.append([])
            
            if self.calculate_fitness(new_solution) < self.calculate_fitness(solution):
                return new_solution
            return solution

        # 50% de chance de Realocação Inteligente (Best Insertion)
        else:
            new_solution = self.inter_route_swap_search(solution)
            return new_solution
            
    def two_opt_local_search(self, solution):
        """Busca local 2-Opt (Intra-Route) para refinar a rota de cada veículo."""
        improved_solution = [r[:] for r in solution]
        total_cost = self.calculate_fitness(improved_solution)

        improved = True
        while improved:
            improved = False
            for route_idx in range(len(improved_solution)):
                vehicle_route = improved_solution[route_idx]
                
                depot_indices = [i for i, x in enumerate(vehicle_route) if x == DEPOT_INDEX]
                
                points_only = [gene for gene in vehicle_route if gene != DEPOT_INDEX]
                
                if len(points_only) < 3:
                    continue
                
                route_changed = False
                
                for i in range(len(points_only) - 1):
                    for j in range(i + 1, len(points_only)):
                        temp_points = points_only[:]
                        temp_points[i:j+1] = temp_points[i:j+1][::-1]

                        temp_route = [DEPOT_INDEX]
                        point_iter = iter(temp_points)
                        
                        for k in range(len(depot_indices) - 1):
                            start_idx = depot_indices[k]
                            end_idx = depot_indices[k+1]
                            num_delivery_points = len(vehicle_route[start_idx:end_idx+1]) - 2
                            
                            for _ in range(num_delivery_points):
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
                            improved = True
                            route_changed = True
                            break
                    if route_changed: break
                if route_changed: break
        
        return improved_solution

    def run(self, epoch_callback: callable = None):
        """Executa o loop do algoritmo genético com otimizações de desempenho."""
        
        cpu_count = multiprocessing.cpu_count()
        chunksize = self.population_size // cpu_count if self.population_size > cpu_count else 1

        fitness_cache = [self.calculate_fitness(sol) for sol in self.population]
        best_idx = min(range(len(fitness_cache)), key=lambda i: fitness_cache[i])
        best_solution = self.population[best_idx]
        best_cost = fitness_cache[best_idx]
        count_generations_without_improvement = 0
        count_generations_without_improvement_for_mutation = 0
        mutation_rate = self.mutation_rate

        for generation in range(self.generations):
            new_population = []
            
            # --- REINICIALIZAÇÃO ---
            if count_generations_without_improvement >= self.max_no_improvement:
                logging.info("Geração %d: Sem melhoria por %d gerações. Reiniciando população.", generation + 1, self.max_no_improvement)
                self.population = self.create_initial_population_hybrid()
                fitness_cache = [self.calculate_fitness(sol) for sol in self.population]
                count_generations_without_improvement = 0
                mutation_rate = self.mutation_rate

            # --- AUMENTO DE MUTAÇÃO ---
            if count_generations_without_improvement_for_mutation >= COUNT_GENERATIONS_WITHOUT_IMPROVEMENT_FOR_MUTATION:
                logging.info("Geração %d: Sem melhoria por %d gerações. Aumentando taxa de mutação.", generation + 1, COUNT_GENERATIONS_WITHOUT_IMPROVEMENT_FOR_MUTATION)
                mutation_rate = min(0.5, mutation_rate * 1.1)
                count_generations_without_improvement_for_mutation = 0

            # --- BUSCA LOCAL INTER-ROTAS (Adição Requisitada) ---
            if generation % COUNT_GENERATIONS_WITHOUT_IMPROVEMENT_FOR_MUTATION == 0 and best_cost != float('inf'):
                logging.info("Geração %d: Aplicando Busca Local Inter-Rotas (Swap).", generation + 1)
                inter_optimized_solution = self.inter_route_swap_search(best_solution)
                inter_optimized_cost = self.calculate_fitness(inter_optimized_solution)
                
                if inter_optimized_cost < best_cost:
                    best_cost = inter_optimized_cost
                    best_solution = inter_optimized_solution
                    count_generations_without_improvement = 0
                    logging.info("Geração %d: Inter-Swap encontrou uma melhoria. Novo custo: %.2f", generation + 1, best_cost)


            # --- BUSCA LOCAL INTRA-ROTAS (2-Opt) ---
            if generation % TWO_OPT_FREQUENCY == 0 and best_cost != float('inf'):
                logging.info("Geração %d: Aplicando busca local Intra-Rotas (2-Opt).", generation + 1)
                local_optimized_solution = self.two_opt_local_search(best_solution)
                local_optimized_cost = self.calculate_fitness(local_optimized_solution)
                
                if local_optimized_cost < best_cost:
                    best_cost = local_optimized_cost
                    best_solution = local_optimized_solution
                    count_generations_without_improvement = 0
                    logging.info("Geração %d: 2-Opt encontrou uma melhoria. Novo custo: %.2f", generation + 1, best_cost)


            # --- SELEÇÃO E GERAÇÃO ---
            best_gen_idx = min(range(len(fitness_cache)), key=lambda i: fitness_cache[i])
            best_of_gen = self.population[best_gen_idx]
            new_population.append(best_of_gen)
            
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents()
                child = self.crossover(parent1, parent2)
                mutated_child = self.mutate(child, mutation_rate)
                new_population.append(mutated_child)

            with multiprocessing.Pool(processes=cpu_count) as pool:
                new_fitness_cache = pool.map(self.calculate_fitness, new_population, chunksize=chunksize)

            self.population = new_population
            fitness_cache = new_fitness_cache
            current_best_idx = min(range(len(fitness_cache)), key=lambda i: fitness_cache[i])
            current_best = self.population[current_best_idx]
            current_best_cost = fitness_cache[current_best_idx]

            logging.info('Geração %d - Custo da melhor solução na população: %.2f', generation + 1, current_best_cost)

            # --- ATUALIZAÇÃO DO MELHOR GLOBAL ---
            if current_best_cost < best_cost:
                best_solution = current_best
                best_cost = current_best_cost
                count_generations_without_improvement = 0
                count_generations_without_improvement_for_mutation = 0
            else:
                count_generations_without_improvement += 1
                count_generations_without_improvement_for_mutation += 1

            logging.info('Geração %d - Melhor Custo Global: %.2f', generation + 1, best_cost)

            # --- CALLBACK ---
            if epoch_callback:
                vehicle_points = []
                for i, route in enumerate(best_solution):
                    if route: 
                        trip_cost, trip_duration, trip_distance = self._get_trip_cost(route)
                        
                        vehicle_points.append({
                            'vehicle_id': i + 1,
                            'points': len([p for p in route if p != 0]), 
                            'route': route,
                            'distance': round(trip_distance / 1000, 2), 
                            'duration': round(trip_duration / 60, 1),   
                            'cost': round(trip_cost, 4) if trip_cost != float('inf') else None
                        })
                
                epoch_callback(
                    epoch=generation + 1,
                    vehicles=len(vehicle_points),
                    vehicle_data=vehicle_points
                )

        return best_solution, best_cost