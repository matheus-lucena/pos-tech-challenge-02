import multiprocessing
import requests
import random
from points import POINTS
import logging
import time
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# --- PARÂMETROS ---
OSRM_BASE_URL = "http://localhost:5000/table/v1/driving/"
POPULATION_SIZE = 300
GENERATIONS = 2000
MUTATION_RATE = 0.05
MAX_VEHICLES = 10
VEHICLE_CAPACITY = 8 # Max number of stops per vehicle
MAX_TRIP_DURATION = 2 * 3600 # X hours in seconds
MAX_TRIP_DISTANCE = 50000 # Max distance in meters
TIME_TO_STOP = 180 # 3 minutes in seconds per stop
TIME_DEPOT_STOP = 180 # 3 minutes in seconds per stop


COUNT_GENERATIONS_WITHOUT_IMPROVEMENT = 20
COUNT_GENERATIONS_WITHOUT_IMPROVEMENT_FOR_MUTATION = 2

DEPOT_INDEX = 0 # Assuming the first point is the depot

def get_matrix(locations):
    coords_str = ";".join([f"{lon},{lat}" for lat, lon in locations])
    url = f"{OSRM_BASE_URL}{coords_str}?sources=all&destinations=all&annotations=duration,distance"

    logging.info('Requesting OSRM matrix: %s', url)

    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error("Erro ao conectar ao OSRM: %s", e)
        return None

# --- FUNÇÃO PARA OBTER A MATRIZ DO OSRM ---
def get_cost_matrix(locations):
    """
    Solicita a matriz de distâncias ao servidor OSRM.
    Retorna uma lista de listas (matriz de tempos em segundos).
    """
    data = get_matrix(locations)
    if data and 'durations' in data and 'distances' in data:
        return data['durations'], data['distances']
    else:
        logging.error("Erro na resposta do OSRM: 'durations/distances' não encontrado.")
        return None, None

# --- CLASSE DO ALGORITMO GENÉTICO VRP ---
class VRPGeneticAlgorithm:
    def __init__(self, duration_matrix, distance_matrix, num_points, time_weight=1, distance_weight=0):
        self.duration_matrix = duration_matrix
        self.distance_matrix = distance_matrix
        self.num_points = num_points
        self.time_weight = time_weight
        self.distance_weight = distance_weight
        self.population = self.create_initial_population()

    def create_initial_population(self):
        population = []
        all_points = list(range(1, self.num_points))

        for _ in range(POPULATION_SIZE):
            random.shuffle(all_points)

            solution = [[] for _ in range(MAX_VEHICLES)]
            
            for i, point in enumerate(all_points):
                vehicle_index = i % MAX_VEHICLES
                solution[vehicle_index].append(point)

            for i in range(MAX_VEHICLES):
                vehicle_route = [DEPOT_INDEX]
                trip_points = 0
                for point in solution[i]:
                    vehicle_route.append(point)
                    trip_points += 1
                    if trip_points == VEHICLE_CAPACITY:
                        vehicle_route.append(DEPOT_INDEX)
                        trip_points = 0
                
                if vehicle_route[-1] != DEPOT_INDEX:
                    vehicle_route.append(DEPOT_INDEX)
                
                solution[i] = vehicle_route
            
            population.append(solution)
        return population

    def calculate_fitness(self, solution):
        total_solution_cost = 0.0
        solution_tuple = tuple(tuple(route) for route in solution)
        for vehicle_route in solution:
            if not vehicle_route:
                continue
            # A rota do veículo é uma sequência de viagens. Vamos processá-las uma a uma.
            depot_indices = [i for i, x in enumerate(vehicle_route) if x == DEPOT_INDEX]
            
            # This check is crucial to identify malformed routes
            if vehicle_route[0] != DEPOT_INDEX or vehicle_route[-1] != DEPOT_INDEX or len(depot_indices) < 2:
                logging.warning('Rota de veículo malformada. Rota: %s', vehicle_route)
                return float('inf')

            for i in range(len(depot_indices) - 1):
                start_idx = depot_indices[i]
                end_idx = depot_indices[i+1]
                trip_points = vehicle_route[start_idx:end_idx+1]
                
                current_trip_duration = 0
                current_trip_distance = 0
                current_trip_stops = len(trip_points) - 2 # -2 para ignorar os depósitos
                
                if current_trip_stops < 0: # Caso de uma rota que é apenas [0, 0]
                    current_trip_stops = 0

                last_point_idx = trip_points[0]
                
                for point_idx in trip_points[1:]:
                    if last_point_idx != -1 and point_idx != -1:
                        current_trip_duration += self.duration_matrix[last_point_idx][point_idx]
                        current_trip_distance += self.distance_matrix[last_point_idx][point_idx]
                    
                    if point_idx != DEPOT_INDEX:
                        current_trip_duration += TIME_TO_STOP
                    else:
                        current_trip_duration += TIME_DEPOT_STOP
                        
                    last_point_idx = point_idx
                
                if current_trip_duration > MAX_TRIP_DURATION or current_trip_distance > MAX_TRIP_DISTANCE or current_trip_stops > VEHICLE_CAPACITY:
                    logging.debug('Viagem inválida. Duração: %d, Distância: %d, Paradas: %d', current_trip_duration, current_trip_distance, current_trip_stops)
                    return float('inf')

                normalized_duration = current_trip_duration / MAX_TRIP_DURATION
                normalized_distance = current_trip_distance / MAX_TRIP_DISTANCE
                trip_cost = (self.time_weight * normalized_duration) + (self.distance_weight * normalized_distance)
                total_solution_cost += trip_cost
        return total_solution_cost

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
        Crossover especializado para VRP que preserva a estrutura de múltiplos
        retornos ao depósito.
        """
        if not parent1 or not parent2:
            return random.choice([parent1, parent2])

        # Coleta todos os pontos de ambos os pais
        parent1_points = [p for route in parent1 for p in route if p != DEPOT_INDEX]
        parent2_points = [p for route in parent2 for p in route if p != DEPOT_INDEX]
        all_unique_points = list(set(parent1_points + parent2_points))
        random.shuffle(all_unique_points)
        
        # Cria um "esqueleto" de rota com a estrutura do parent1
        child = []
        points_to_assign = all_unique_points[:]
        
        for parent_route in parent1:
            if not parent_route:
                continue

            # Conta o número de viagens e paradas na rota do pai
            num_trips = len([p for p in parent_route if p == DEPOT_INDEX]) - 1
            trips_stops = []
            
            # Divide a rota do pai em viagens individuais e conta as paradas
            depot_indices = [i for i, x in enumerate(parent_route) if x == DEPOT_INDEX]
            for i in range(len(depot_indices) - 1):
                trip_points = parent_route[depot_indices[i]:depot_indices[i+1]+1]
                trips_stops.append(len(trip_points) - 2)

            # Reconstroi a rota do filho com os novos pontos, respeitando a estrutura do pai
            child_route = [DEPOT_INDEX]
            
            for num_stops in trips_stops:
                current_trip_stops = 0
                while current_trip_stops < num_stops and points_to_assign:
                    child_route.append(points_to_assign.pop(0))
                    current_trip_stops += 1
                child_route.append(DEPOT_INDEX)
            
            # Adiciona a rota do filho na solução
            child.append(child_route)

        # Adiciona os pontos restantes que não couberam na estrutura do pai
        if points_to_assign:
            temp_points = points_to_assign[:]
            # Tenta preencher rotas existentes que ainda não estão cheias
            for route in child:
                current_stops = len([p for p in route if p != DEPOT_INDEX])
                while current_stops < VEHICLE_CAPACITY and temp_points:
                    route.insert(len(route) - 1, temp_points.pop(0))
                    current_stops += 1
            
            # Cria novas rotas para os pontos que ainda restaram
            while temp_points:
                new_route = [DEPOT_INDEX]
                current_stops = 0
                while current_stops < VEHICLE_CAPACITY and temp_points:
                    new_route.append(temp_points.pop(0))
                    current_stops += 1
                new_route.append(DEPOT_INDEX)
                child.append(new_route)

        # Garante que o número de veículos não excede o máximo
        while len(child) < MAX_VEHICLES:
            child.append([])
        
        return child

    def mutate(self, solution, mutation_rate):
        """
        Operador de mutação híbrido. 
        Com 50% de chance, faz uma mutação de inversão (2-Opt).
        Com 50% de chance, faz uma mutação de realocação.
        """
        is_mutated = random.random() < mutation_rate
        if not is_mutated:
            return solution

        if random.random() < 0.5:
            # Mutação de Inversão (2-Opt)
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
                    # This is a safeguard against an unexpected point count mismatch.
                    # It should not be triggered if the code is correct.
                    break 

            return new_solution

        else:
            # Mutação de Realocação de Pontos
            points_only = [gene for route in solution for gene in route if gene != DEPOT_INDEX]
            if len(solution) > 1 and len(points_only) > 1:
                random_point_to_move = random.choice(points_only)
                
                source_route_index = -1
                point_in_route_index = -1
                for idx, route in enumerate(solution):
                    if random_point_to_move in route:
                        source_route_index = idx
                        point_in_route_index = route.index(random_point_to_move)
                        break
                
                target_route_index = random.choice([i for i in range(len(solution)) if i != source_route_index])
                
                if source_route_index != -1 and target_route_index != -1:
                    new_solution = [r[:] for r in solution]
                    new_solution[source_route_index].pop(point_in_route_index)
                    # Handle case where a trip becomes empty
                    if len(new_solution[source_route_index]) == 2 and new_solution[source_route_index][0] == new_solution[source_route_index][1]:
                        new_solution[source_route_index] = []
                    target_route_length = len(new_solution[target_route_index])
                    insert_index = random.randint(1, target_route_length - 1)
                    new_solution[target_route_index].insert(insert_index, random_point_to_move)
                    return new_solution
        
        return solution

    def two_opt_local_search(self, solution):
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
                        # Crie uma cópia da rota para testar a mutação
                        temp_points = points_only[:]
                        temp_points[i:j+1] = temp_points[i:j+1][::-1]

                        # Crie uma solução temporária completa para avaliar o custo
                        temp_solution = [r[:] for r in improved_solution]
                        temp_route = [DEPOT_INDEX] + temp_points + [DEPOT_INDEX]
                        temp_solution[route_idx] = temp_route

                        # Calcule o custo da nova solução completa
                        new_cost = self.calculate_fitness(temp_solution)

                        if new_cost < total_cost:
                            improved_solution = temp_solution
                            total_cost = new_cost
                            improved = True
                            break # Reinicie a busca com a nova rota melhorada
                    if improved:
                        break
                if improved:
                    break
        return improved_solution

    def run(self):
        """Executa o loop do algoritmo genético com otimizações de desempenho."""
        fitness_cache = [self.calculate_fitness(sol) for sol in self.population]
        best_idx = min(range(len(fitness_cache)), key=lambda i: fitness_cache[i])
        best_solution = self.population[best_idx]
        best_cost = fitness_cache[best_idx]
        count_generations_without_improvement = 0
        count_generations_without_improvement_for_mutation = 0
        mutation_rate = MUTATION_RATE

        for generation in range(GENERATIONS):
            new_population = []

            # Aplica a busca local na melhor solução a cada 100 gerações
            if generation % 100 == 0 and best_cost != float('inf'):
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
            new_fitness_cache = [fitness_cache[best_gen_idx]]

            if count_generations_without_improvement > COUNT_GENERATIONS_WITHOUT_IMPROVEMENT:
                mutation_rate = min(0.5, mutation_rate * 1.05)
                count_generations_without_improvement = 0
                count_generations_without_improvement_for_mutation += 1
                logging.info('Nova taxa de mutação: %s', mutation_rate)
            if count_generations_without_improvement_for_mutation > COUNT_GENERATIONS_WITHOUT_IMPROVEMENT_FOR_MUTATION:
                break

            while len(new_population) < POPULATION_SIZE:
                parent1, parent2 = self.select_parents()
                child = self.crossover(parent1, parent2)
                mutated_child = self.mutate(child, mutation_rate)
                new_population.append(mutated_child)
                # new_fitness_cache.append(self.calculate_fitness(mutated_child))

            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                new_fitness_cache = pool.map(self.calculate_fitness, new_population)

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
        
        print("\n--- Resultado Final ---")
        
        print(f"Melhor custo de solução encontrado: {best_cost:.2f} (em unidades normalizadas)")
        print(f"Número de veículos utilizados: {len(best_solution)}")
        
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
                total_travel_duration = 0
                total_travel_distance = 0
                
                if travel[0] != DEPOT_INDEX:
                    continue

                last_point_idx = travel[0]
                total_travel_duration += TIME_DEPOT_STOP
                
                for point_idx in travel[1:]:
                    total_travel_duration += duration_matrix[last_point_idx][point_idx]
                    total_travel_distance += distance_matrix[last_point_idx][point_idx]
                    
                    if point_idx != DEPOT_INDEX:
                        total_travel_duration += TIME_TO_STOP
                    else:
                        total_travel_duration += TIME_DEPOT_STOP
                        
                    last_point_idx = point_idx
                
                print(f"      Viagem {j+1}:")
                print(f"        Paradas: {travel[1:-1]}")
                print(f"        Total de paradas de entrega: {len(travel) - 2}")
                print(f"        Tempo de viagem: {total_travel_duration / 60:.2f} minutos")
                print(f"        Distância de viagem: {total_travel_distance:.2f} metros")
                print(f"        Rota (índices): {travel}")


    end_time = time.time()
    logging.info("Tempo total de execução: %.2f segundos", end_time - start_time)