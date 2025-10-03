import multiprocessing
import requests
import random
import math
from points import POINTS
from llmintegration import gerar_pdf_relatorio
import logging
import time
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# --- PARÂMETROS ---
OSRM_BASE_URL = "http://localhost:5000/table/v1/driving/"
POPULATION_SIZE = 400
POPULATION_HEURISTIC_TAX = 0.5
GENERATIONS = 20
MUTATION_RATE = 0.05
MAX_VEHICLES = 20
VEHICLE_MAX_POINTS = 8 # Max number of stops per vehicle
MAX_TRIP_DURATION = 8 * 3600 # X hours in seconds
MAX_TRIP_DISTANCE = 50000 # Max distance in meters
TIME_TO_STOP = 180 # 3 minutes in seconds per stop
TIME_DEPOT_STOP = 180 # 3 minutes in seconds per stop
TWO_OPT_FREQUENCY = 100 # Apply local search every X generations

COUNT_GENERATIONS_WITHOUT_IMPROVEMENT = 50
COUNT_GENERATIONS_WITHOUT_IMPROVEMENT_FOR_MUTATION = 5

DEPOT_INDEX = 0 # Assuming the first point is the depot

def generate_json_output(best_solution, best_cost, duration_matrix, distance_matrix, POINTS):
    """Monta a estrutura JSON da solução VRP final."""
    all_routes_data = []
    
    for i, route in enumerate(best_solution):
        if not route:
            continue

        travels_data = []
        travels = []
        current_travel = []
        
        # Divide a rota do veículo em viagens (Depot -> ... -> Depot)
        for point_idx in route:
            current_travel.append(point_idx)
            if point_idx == DEPOT_INDEX and len(current_travel) > 1:
                travels.append(current_travel)
                current_travel = [DEPOT_INDEX]
        
        if len(current_travel) > 1:
            travels.append(current_travel)
        
        total_route_duration = 0
        total_route_distance = 0
        
        # Calcula os detalhes de cada viagem
        for j, travel in enumerate(travels):
            
            # Recálculo das métricas (usa a lógica que estava na seção de impressão)
            current_trip_duration = 0
            current_trip_distance = 0
            last_point_idx = travel[0]
            
            current_trip_duration += TIME_DEPOT_STOP # Tempo inicial no depósito
            
            for point_idx in travel[1:]:
                # Se for um ponto intermediário de viagem/rota
                if last_point_idx != -1 and point_idx != -1:
                    current_trip_duration += duration_matrix[last_point_idx][point_idx]
                    current_trip_distance += distance_matrix[last_point_idx][point_idx]
                
                # Tempo de parada
                if point_idx != DEPOT_INDEX:
                    current_trip_duration += TIME_TO_STOP
                else:
                    current_trip_duration += TIME_DEPOT_STOP
                        
                last_point_idx = point_idx
                
            total_route_duration += current_trip_duration
            total_route_distance += current_trip_distance
            
            formatted_travel_points = "/".join([f"{POINTS[idx][0]},{POINTS[idx][1]}" for idx in travel])
            
            travels_data.append({
                "trip_id": j + 1,
                "stop_indices": travel[1:-1],
                "total_stops": len(travel) - 2,
                "duration_minutes": current_trip_duration / 60,
                "distance_meters": current_trip_distance,
                "route_coordinates": formatted_travel_points
            })

        all_routes_data.append({
            "vehicle_id": i + 1,
            "total_route_duration_minutes": total_route_duration / 60,
            "total_route_distance_meters": total_route_distance,
            "full_route_indices": route,
            "full_route_coordinates": "/".join([f"{POINTS[idx][0]},{POINTS[idx][1]}" for idx in route]),
            "trips": travels_data
        })

    final_output = {
        "best_normalized_cost": best_cost,
        "number_of_vehicles_used": len(all_routes_data),
        "cost_matrix_based_on": OSRM_BASE_URL,
        "base_constraints": {
            "max_fleet_vehicles": MAX_VEHICLES,
            "max_stops_per_trip": VEHICLE_MAX_POINTS,
            "max_duration_minutes": MAX_TRIP_DURATION / 60
        },
        "routes": all_routes_data
    }

    gerar_pdf_relatorio(final_output)  # Chama a função para gerar o PDF com o dicionário JSON
    return final_output

def get_matrix(locations):
    coords_str = ";".join([f"{lon},{lat}" for lat, lon in locations])
    url = f"{OSRM_BASE_URL}{coords_str}?sources=all&destinations=all&annotations=duration,distance"

    # logging.info('Requesting OSRM matrix: %s', url) # Comentado para não poluir o log

    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error("Erro ao conectar ao OSRM: %s", e)
        return None

def get_cost_matrix(locations):
    data = get_matrix(locations)
    if data and 'durations' in data and 'distances' in data:
        return data['durations'], data['distances']
    else:
        logging.error("Erro na resposta do OSRM: 'durations/distances' não encontrado.")
        return None, None

# --- CLASSE DO ALGORITMO GENÉTICO VRP ---
class VRPGeneticAlgorithm:
    def __init__(self, duration_matrix, distance_matrix, num_points, time_weight=0.5, distance_weight=0.5):
        self.duration_matrix = duration_matrix
        self.distance_matrix = distance_matrix
        self.num_points = num_points
        self.time_weight = time_weight
        self.distance_weight = distance_weight
        self.points_coordinates = POINTS # <--- CORREÇÃO: PONTOS AGORA ACESSÍVEIS
        self.population = self.create_initial_population_hybrid() # Alterado para híbrido

    # --- MÉTODOS DE GERAÇÃO DE POPULAÇÃO ---

    def _create_initial_population_random_only(self):
        """Método original (Round Robin) para gerar a parte aleatória da população."""
        all_points = list(range(1, self.num_points))
        random.shuffle(all_points)

        solution = [[] for _ in range(MAX_VEHICLES)]
        
        for i, point in enumerate(all_points):
            vehicle_index = i % MAX_VEHICLES
            solution[vehicle_index].append(point)

        final_solution = []
        for route_points in solution:
            vehicle_route = [DEPOT_INDEX]
            trip_points = 0
            for point in route_points:
                vehicle_route.append(point)
                trip_points += 1
                if trip_points == VEHICLE_MAX_POINTS:
                    vehicle_route.append(DEPOT_INDEX)
                    trip_points = 0
            
            if vehicle_route and vehicle_route[-1] != DEPOT_INDEX:
                vehicle_route.append(DEPOT_INDEX)
            
            if len(vehicle_route) > 2:
                final_solution.append(vehicle_route)
            else:
                final_solution.append([])
                
        # Garante que o tamanho do cromossomo seja MAX_VEHICLES
        final_solution = final_solution[:MAX_VEHICLES]
        while len(final_solution) < MAX_VEHICLES:
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
        
        while temp_points_list and len(final_solution) < MAX_VEHICLES:
            
            cluster_points = temp_points_list[:VEHICLE_MAX_POINTS]
            temp_points_list = temp_points_list[VEHICLE_MAX_POINTS:]

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
        final_solution = final_solution[:MAX_VEHICLES]
        while len(final_solution) < MAX_VEHICLES:
            final_solution.append([])
            
        random.shuffle(final_solution)
        
        return final_solution

    def create_initial_population_hybrid(self):
        num_clustered = int(POPULATION_SIZE * POPULATION_HEURISTIC_TAX)
        num_random = POPULATION_SIZE - num_clustered
        
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
        """Calcula o custo normalizado e a viabilidade de uma única viagem [0, p1, ..., 0]."""
        if len(trip_points) < 2 or trip_points[0] != DEPOT_INDEX or trip_points[-1] != DEPOT_INDEX:
            return float('inf'), 0, 0, 0

        current_trip_duration = 0
        current_trip_distance = 0
        current_trip_stops = len(trip_points) - 2

        last_point_idx = trip_points[0]
        current_trip_duration += TIME_DEPOT_STOP
        
        for point_idx in trip_points[1:]:
            current_trip_duration += self.duration_matrix[last_point_idx][point_idx]
            current_trip_distance += self.distance_matrix[last_point_idx][point_idx]
            
            if point_idx != DEPOT_INDEX:
                current_trip_duration += TIME_TO_STOP
            elif point_idx == DEPOT_INDEX and last_point_idx != DEPOT_INDEX:
                current_trip_duration += TIME_DEPOT_STOP
                
            last_point_idx = point_idx

        # Penalidade infinita para inviabilidade (Hard Constraint)
        if current_trip_duration > MAX_TRIP_DURATION or current_trip_distance > MAX_TRIP_DISTANCE or current_trip_stops > VEHICLE_MAX_POINTS:
            return float('inf'), 0, 0, 0
        
        # Cálculo do custo normalizado (Fitness)
        normalized_duration = current_trip_duration / MAX_TRIP_DURATION
        normalized_distance = current_trip_distance / MAX_TRIP_DISTANCE
        trip_cost = (self.time_weight * normalized_duration) + (self.distance_weight * normalized_distance)
        
        return trip_cost, current_trip_duration, current_trip_distance, current_trip_stops

    def calculate_fitness(self, solution):
        """
        Calcula a aptidão (custo) da solução total. Inclui as Restrições Rígidas de Frota e Cobertura.
        """
        total_solution_cost = 0.0
        
        # --- RESTRIÇÃO RÍGIDA 1: LIMITE MÁXIMO DE VEÍCULOS ---
        vehicles_used = len([r for r in solution if r])
        
        if vehicles_used > MAX_VEHICLES:
             return float('inf')
        
        # --- RESTRIÇÃO RÍGIDA 2 & 3: VIABILIDADE E COBERTURA ---
        
        visited_points = set()
        
        for vehicle_route in solution:
            if not vehicle_route:
                continue
                
            depot_indices = [i for i, x in enumerate(vehicle_route) if x == DEPOT_INDEX]
            
            if vehicle_route[0] != DEPOT_INDEX or vehicle_route[-1] != DEPOT_INDEX or len(depot_indices) < 2:
                # O warning aqui é útil para debugging, mas o retorno é float('inf')
                # logging.warning('Rota de veículo malformada. Rota: %s', vehicle_route)
                return float('inf')

            for i in range(len(depot_indices) - 1):
                start_idx = depot_indices[i]
                end_idx = depot_indices[i+1]
                trip_points = vehicle_route[start_idx:end_idx+1]
                
                trip_cost, _, _, _ = self._get_trip_cost(trip_points)
                
                if trip_cost == float('inf'):
                    return float('inf')

                total_solution_cost += trip_cost
                
                visited_points.update(p for p in trip_points if p != DEPOT_INDEX)
        
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
        Crossover VRP com limite de MAX_VEHICLES e garantia de tamanho do cromossomo.
        """
        if not parent1 or not parent2:
            return random.choice([parent1, parent2])

        parent1_points = [p for route in parent1 for p in route if p != DEPOT_INDEX]
        parent2_points = [p for route in parent2 for p in route if p != DEPOT_INDEX]
        all_unique_points = list(set(parent1_points + parent2_points))
        random.shuffle(all_unique_points)
        
        child = []
        points_to_assign = all_unique_points[:]
        
        for parent_route in parent1:
            if not parent_route or len(child) >= MAX_VEHICLES:
                break

            depot_indices = [i for i, x in enumerate(parent_route) if x == DEPOT_INDEX]
            if len(depot_indices) < 2: continue
                
            trips_stops = []
            for i in range(len(depot_indices) - 1):
                trip_points = parent_route[depot_indices[i]:depot_indices[i+1]+1]
                trips_stops.append(len(trip_points) - 2)

            child_route = [DEPOT_INDEX]
            
            for num_stops in trips_stops:
                current_trip_stops = 0
                while current_trip_stops < num_stops and points_to_assign:
                    child_route.append(points_to_assign.pop(0))
                    current_trip_stops += 1
                child_route.append(DEPOT_INDEX)
            
            child.append(child_route)

        if points_to_assign:
            temp_points = points_to_assign[:]
            
            for route in child:
                current_stops = len([p for p in route if p != DEPOT_INDEX])
                while current_stops < VEHICLE_MAX_POINTS and temp_points:
                    route.insert(len(route) - 1, temp_points.pop(0))
                    current_stops += 1
            
            while temp_points and len(child) < MAX_VEHICLES:
                new_route = [DEPOT_INDEX]
                current_stops = 0
                while current_stops < VEHICLE_MAX_POINTS and temp_points:
                    new_route.append(temp_points.pop(0))
                    current_stops += 1
                new_route.append(DEPOT_INDEX)
                child.append(new_route)

        child = child[:MAX_VEHICLES]
        while len(child) < MAX_VEHICLES:
            child.append([])
            
        return child

    def mutate(self, solution, mutation_rate):
        """
        Operador de mutação híbrido: 50% Inversão (2-Opt) / 50% Realocação Inteligente (Best Insertion).
        """
        is_mutated = random.random() < mutation_rate
        if not is_mutated:
            return solution

        # 50% de chance de 2-Opt
        if random.random() < 0.5:
            # Mutação de Inversão (2-Opt) - Lógica inalterada
            points_only = [gene for route in solution for gene in route if gene != DEPOT_INDEX]
            if len(points_only) >= 2:
                idx1, idx2 = sorted(random.sample(range(len(points_only)), 2))
                points_only[idx1:idx2] = points_only[idx1:idx2][::-1]

            new_solution = []
            points_iter = iter(points_only)
            
            for route in solution:
                if not route:
                    new_solution.append([])
                    continue
                reconstructed_route = [DEPOT_INDEX]
                
                try:
                    num_delivery_points = len([gene for gene in route if gene != DEPOT_INDEX])
                    for _ in range(num_delivery_points):
                        reconstructed_route.append(next(points_iter))
                    reconstructed_route.append(DEPOT_INDEX)
                    new_solution.append(reconstructed_route)
                except StopIteration:
                    break 

            while len(new_solution) < MAX_VEHICLES:
                new_solution.append([])
            return new_solution

        # 50% de chance de Realocação Inteligente (Best Insertion)
        else:
            points_only = [gene for route in solution for gene in route if gene != DEPOT_INDEX]
            
            if len([r for r in solution if r]) < 2 or not points_only:
                return solution 
            
            random_point_to_move = random.choice(points_only)
            
            source_route_index = -1
            point_in_route_index = -1
            for idx, route in enumerate(solution):
                if random_point_to_move in route:
                    source_route_index = idx
                    point_in_route_index = route.index(random_point_to_move)
                    break
            
            best_target_route_index = -1
            best_insert_index = -1
            min_new_cost = float('inf')
            
            original_cost = self.calculate_fitness(solution)
            
            # Busca a Melhor Inserção (Best Insertion)
            for target_route_index in range(len(solution)):
                if target_route_index == source_route_index:
                    continue
                
                target_route = solution[target_route_index]
                
                if not target_route:
                    insert_positions = [1]
                else:
                    insert_positions = range(1, len(target_route))
                
                for insert_index in insert_positions:
                    
                    temp_solution = [r[:] for r in solution]
                    
                    # A. Remove o ponto da rota de origem
                    temp_solution[source_route_index].pop(point_in_route_index)
                    
                    if len(temp_solution[source_route_index]) <= 2 and all(p == DEPOT_INDEX for p in temp_solution[source_route_index]):
                        temp_solution[source_route_index] = []
                    
                    # B. Insere o ponto na nova rota (respeita o limite de veículos)
                    if not temp_solution[target_route_index]:
                        if len([r for r in temp_solution if r]) < MAX_VEHICLES:
                            temp_solution[target_route_index] = [DEPOT_INDEX, random_point_to_move, DEPOT_INDEX]
                        else:
                            continue 
                    else:
                        temp_solution[target_route_index].insert(insert_index, random_point_to_move)

                    # C. Calcula o custo da nova solução (checa a viabilidade)
                    new_cost = self.calculate_fitness(temp_solution)

                    # D. Se a nova solução for a melhor VIÁVEL, armazena
                    if new_cost < min_new_cost:
                        min_new_cost = new_cost
                        best_target_route_index = target_route_index
                        best_insert_index = insert_index
            
            # 3. Aplica a MELHOR realocação
            if min_new_cost < original_cost and best_target_route_index != -1:
                new_solution = [r[:] for r in solution]
                
                # A. Remove
                new_solution[source_route_index].pop(point_in_route_index)
                if len(new_solution[source_route_index]) <= 2 and all(p == DEPOT_INDEX for p in new_solution[source_route_index]):
                    new_solution[source_route_index] = []
                
                # B. Insere na melhor posição
                if not new_solution[best_target_route_index]:
                    new_solution[best_target_route_index] = [DEPOT_INDEX, random_point_to_move, DEPOT_INDEX]
                else:
                    new_solution[best_target_route_index].insert(best_insert_index, random_point_to_move)
                    
                # Garante o tamanho do cromossomo
                while len(new_solution) < MAX_VEHICLES:
                    new_solution.append([])
                    
                return new_solution
                
        return solution 

    def two_opt_local_search(self, solution):
        """Busca local 2-Opt (Intra-Route) para refinar a rota de cada veículo."""
        improved_solution = [r[:] for r in solution]
        total_cost = self.calculate_fitness(improved_solution)

        improved = True
        while improved:
            improved = False
            for route_idx in range(len(improved_solution)):
                vehicle_route = improved_solution[route_idx]
                points_only = [gene for gene in vehicle_route if gene != DEPOT_INDEX]
                
                if len(points_only) < 3:
                    continue
                
                for i in range(len(points_only) - 1):
                    for j in range(i + 1, len(points_only)):
                        temp_points = points_only[:]
                        temp_points[i:j+1] = temp_points[i:j+1][::-1]

                        temp_solution = [r[:] for r in improved_solution]
                        # Simplifica para uma única viagem para aplicar o 2-Opt
                        temp_route = [DEPOT_INDEX] + temp_points + [DEPOT_INDEX]
                        temp_solution[route_idx] = temp_route

                        new_cost = self.calculate_fitness(temp_solution)

                        if new_cost < total_cost:
                            improved_solution = temp_solution
                            total_cost = new_cost
                            improved = True
                            break 
                    if improved:
                        break
                if improved:
                    break
        return improved_solution

    def run(self):
        """Executa o loop do algoritmo genético com otimizações de desempenho."""
        
        cpu_count = multiprocessing.cpu_count()
        chunksize = POPULATION_SIZE // cpu_count if POPULATION_SIZE > cpu_count else 1

        fitness_cache = [self.calculate_fitness(sol) for sol in self.population]
        best_idx = min(range(len(fitness_cache)), key=lambda i: fitness_cache[i])
        best_solution = self.population[best_idx]
        best_cost = fitness_cache[best_idx]
        count_generations_without_improvement = 0
        count_generations_without_improvement_for_mutation = 0
        mutation_rate = MUTATION_RATE

        for generation in range(GENERATIONS):
            new_population = []
            
            if count_generations_without_improvement >= COUNT_GENERATIONS_WITHOUT_IMPROVEMENT:
                logging.info("Geração %d: Sem melhoria por %d gerações. Reiniciando população.", generation + 1, COUNT_GENERATIONS_WITHOUT_IMPROVEMENT)
                self.population = self.create_initial_population_hybrid()
                fitness_cache = [self.calculate_fitness(sol) for sol in self.population]
                count_generations_without_improvement = 0
                mutation_rate = MUTATION_RATE

            if count_generations_without_improvement_for_mutation >= COUNT_GENERATIONS_WITHOUT_IMPROVEMENT_FOR_MUTATION:
                logging.info("Geração %d: Sem melhoria por %d gerações. Aumentando taxa de mutação.", generation + 1, COUNT_GENERATIONS_WITHOUT_IMPROVEMENT_FOR_MUTATION)
                mutation_rate = min(0.5, mutation_rate * 1.1)
                count_generations_without_improvement_for_mutation = 0

            # Aplica a busca local na melhor solução a cada 100 gerações
            if generation % TWO_OPT_FREQUENCY == 0 and best_cost != float('inf'):
                logging.info("Geração %d: Aplicando busca local na melhor solução.", generation + 1)
                local_optimized_solution = self.two_opt_local_search(best_solution)
                local_optimized_cost = self.calculate_fitness(local_optimized_solution)
                
                if local_optimized_cost < best_cost:
                    best_cost = local_optimized_cost
                    best_solution = local_optimized_solution
                    count_generations_without_improvement = 0
                    logging.info("Geração %d: Busca local encontrou uma melhoria. Novo custo: %.2f", generation + 1, best_cost)

            best_gen_idx = min(range(len(fitness_cache)), key=lambda i: fitness_cache[i])
            best_of_gen = self.population[best_gen_idx]
            new_population.append(best_of_gen)
            
            while len(new_population) < POPULATION_SIZE:
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

            if current_best_cost < best_cost:
                best_solution = current_best
                best_cost = current_best_cost
                count_generations_without_improvement = 0
                count_generations_without_improvement_for_mutation = 0
            else:
                count_generations_without_improvement += 1
                count_generations_without_improvement_for_mutation += 1

            logging.info('Geração %d - Melhor Custo Global: %.2f', generation + 1, best_cost)

        return best_solution, best_cost

# --- EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    start_time = time.time()
    duration_matrix, distance_matrix = get_cost_matrix(POINTS)
    
    if duration_matrix and distance_matrix:
        print("Matriz de distâncias recebida com sucesso.")
        
        print(f"\n2. Executando Algoritmo Genético com {MAX_VEHICLES} veículos...")
        ga = VRPGeneticAlgorithm(duration_matrix, distance_matrix, len(POINTS))
        best_solution, best_cost = ga.run()

        final_json_data = generate_json_output(
            best_solution, 
            best_cost, 
            duration_matrix, 
            distance_matrix, 
            POINTS
        )

        json_output = json.dumps(final_json_data, indent=4, ensure_ascii=False)
        
        print("\n--- INÍCIO DA SAÍDA JSON PARA LLM ---")
        print(json_output)
        print("--- FIM DA SAÍDA JSON PARA LLM ---")
        print("\n--- Resultado Final ---")
        
        print(f"Melhor custo de solução encontrado: {best_cost:.2f} (em unidades normalizadas)")
        vehicles_used = len([r for r in best_solution if r])
        print(f"Número de veículos utilizados: {vehicles_used}")
        
        for i, route in enumerate(best_solution):
            if not route:
                continue

            travels = []
            current_travel = []
            for point_idx in route:
                current_travel.append(point_idx)
                if point_idx == DEPOT_INDEX and len(current_travel) > 1:
                    travels.append(current_travel)
                    current_travel = [DEPOT_INDEX]
            
            if len(current_travel) > 1:
                travels.append(current_travel)
            
            print(f"\nRota do Veículo {i+1}:")
            
            for j, travel in enumerate(travels):
                
                trip_cost, total_travel_duration, total_travel_distance, stops = ga._get_trip_cost(travel)
                
                if total_travel_duration == 0:
                    continue
                
                print(f"     Viagem {j+1}:")
                print(f"       Paradas: {travel[1:-1]}")
                print(f"       Total de paradas de entrega: {stops}")
                print(f"       Tempo de viagem: {total_travel_duration / 60:.2f} minutos")
                print(f"       Distância de viagem: {total_travel_distance:.2f} metros")
                print(f"       Rota (índices): {travel}")
                
                formatted_travel_points = "/".join([f"{POINTS[idx][0]},{POINTS[idx][1]}" for idx in travel])
                print(f"       String formatada das paradas da viagem: {formatted_travel_points}")

            formatted_route_points = "/".join([f"{POINTS[idx][0]},{POINTS[idx][1]}" for idx in route])
            print(f"   String formatada das paradas da rota do veículo: {formatted_route_points}")

    end_time = time.time()
    logging.info("Tempo total de execução: %.2f segundos", end_time - start_time)

print(POINTS[0])