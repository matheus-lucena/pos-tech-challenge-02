import requests
import random
from points import POINTS
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# --- PARÂMETROS ---
OSRM_BASE_URL = "http://localhost:5000/table/v1/driving/"
POPULATION_SIZE = 50
GENERATIONS = 100
MUTATION_RATE = 0.05
MAX_VEHICLES = 10
VEHICLE_CAPACITY = 15 # Max number of stops per vehicle
MAX_TRIP_DURATION = 3 * 3600 # 3 hours in seconds
MAX_TRIP_DISTANCE = 50000 # Max distance in meters
TIME_TO_STOP = 180 # 3 minutes in seconds per stop
TIME_DEPOT_STOP = 180 # 3 minutes in seconds per stop

DEPOT_INDEX = 0 # Assuming the first point is the depot

def get_matrix(locations):
    coords_str = ";".join([f"{lon},{lat}" for lat, lon in locations])
    url = f"{OSRM_BASE_URL}{coords_str}?sources=all&destinations=all&annotations=duration,distance"

    logging.info('Requesting OSRM matrix: %s', url)

    response = requests.get(url)
    response.raise_for_status()
    
    return response.json()

# --- FUNÇÃO PARA OBTER A MATRIZ DO OSRM ---
def get_cost_matrix(locations):
    """
    Solicita a matriz de distâncias ao servidor OSRM.
    Retorna uma lista de listas (matriz de tempos em segundos).
    """
    try:
        data = get_matrix(locations)
        if 'durations' in data and 'distances' in data:
            return data['durations'], data['distances']
        else:
            logging.error("Erro na resposta do OSRM: 'durations/distances' não encontrado.")
            return None, None
    except requests.exceptions.RequestException as e:
        logging.error("Erro ao conectar ao OSRM: %s", e)
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
            
            points_to_assign = all_points[:]
            
            # Distribute points into routes for each vehicle
            vehicle_idx = 0
            while points_to_assign:
                route_for_vehicle = []
                current_trip_stops = 0

                # Start a new trip from the depot
                route_for_vehicle.append(DEPOT_INDEX)
                
                # Build the trip until capacity is reached or no more points
                while points_to_assign and current_trip_stops < VEHICLE_CAPACITY:
                    point_to_add = points_to_assign.pop(0)
                    route_for_vehicle.append(point_to_add)
                    current_trip_stops += 1
                
                # If the trip has stops, end it by returning to the depot
                if len(route_for_vehicle) > 1:
                    route_for_vehicle.append(DEPOT_INDEX)

                solution[vehicle_idx].extend(route_for_vehicle)
                
                # Switch to the next vehicle for the next trip
                vehicle_idx = (vehicle_idx + 1) % MAX_VEHICLES
            
            # Clean up empty routes
            solution = [route for route in solution if route]
            
            population.append(solution)
        return population

    def calculate_fitness(self, solution):
        total_solution_cost = 0.0      
        for vehicle_route in solution:
            if not vehicle_route:
                continue

            current_trip_duration = 0
            current_trip_distance = 0
            current_trip_stops = 0
            last_point_idx = DEPOT_INDEX

            for point_idx in vehicle_route:
                if last_point_idx == DEPOT_INDEX:
                    current_trip_duration += TIME_DEPOT_STOP
                else:
                    current_trip_duration += TIME_TO_STOP
                # Check if adding the next point exceeds any trip limit
                # We calculate this first to decide if a new trip is needed
                
                # Duration and distance from the last point of the current trip
                duration_to_next = self.duration_matrix[last_point_idx][point_idx]
                distance_to_next = self.distance_matrix[last_point_idx][point_idx]
                
                # Duration and distance back to the depot from the current point
                duration_back_to_depot = self.duration_matrix[point_idx][DEPOT_INDEX]
                distance_back_to_depot = self.distance_matrix[point_idx][DEPOT_INDEX]
                
                # --- Trip Termination Logic ---
                # If the current trip would exceed limits by adding the next stop,
                # we simulate a return to the depot and start a new trip.
                if (current_trip_duration + duration_to_next + duration_back_to_depot > MAX_TRIP_DURATION or
                    current_trip_distance + distance_to_next + distance_back_to_depot > MAX_TRIP_DISTANCE or
                    current_trip_stops + 1 > VEHICLE_CAPACITY):
                    
                    # Normalize and add the cost of the *just finished* trip
                    normalized_duration = current_trip_duration / MAX_TRIP_DURATION
                    normalized_distance = current_trip_distance / MAX_TRIP_DISTANCE
                    trip_cost = (self.time_weight * normalized_duration) + (self.distance_weight * normalized_distance)
                    total_solution_cost += trip_cost

                    # Reset counters for the new trip
                    current_trip_duration = 0
                    current_trip_distance = 0
                    current_trip_stops = 0
                    last_point_idx = DEPOT_INDEX

                # Continue adding to the current trip
                current_trip_duration += duration_to_next
                current_trip_distance += distance_to_next
                current_trip_stops += 1
                last_point_idx = point_idx

            if(current_trip_duration > MAX_TRIP_DURATION):
                logging.warning('Exceeded max trip duration in fitness calculation.')
                return float('inf')

            if(current_trip_stops > VEHICLE_CAPACITY):
                logging.warning('Exceeded max trip stops in fitness calculation.')
                return float('inf')

            if(current_trip_distance > MAX_TRIP_DISTANCE):
                logging.warning('Exceeded max trip distance in fitness calculation.')
                return float('inf')

            # --- Add cost of the final trip ---
            # Normalize and add the cost of the last trip in the route
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
        """Crossover para VRP que funciona em uma única lista e mantém a estrutura."""
        
        # Flatten parent routes, but exclude depot indices
        parent1_points = [gene for route in parent1 for gene in route if gene != DEPOT_INDEX]
        parent2_points = [gene for route in parent2 for gene in route if gene != DEPOT_INDEX]
        
        child_points = [None] * len(parent1_points)
        
        # Select a segment for crossover
        start, end = sorted(random.sample(range(len(parent1_points)), 2))
        
        # Copy the segment from parent1
        child_points[start:end] = parent1_points[start:end]
        
        # Fill the rest from parent2, skipping duplicates
        current_idx = end
        for gene in parent2_points:
            if gene not in child_points[start:end] and gene not in child_points:
                child_points[current_idx % len(parent1_points)] = gene
                current_idx += 1
        
        # Reconstruct child routes with depot indices at trip boundaries
        child = [[] for _ in range(MAX_VEHICLES)]
        point_idx = 0
        for i in range(MAX_VEHICLES):
            route_len = len([gene for gene in parent1[i] if gene != DEPOT_INDEX])
            if route_len == 0:
                continue
            # Start with depot
            route = [DEPOT_INDEX]
            # Add delivery points
            route += child_points[point_idx:point_idx + route_len]
            point_idx += route_len
            # End with depot
            route.append(DEPOT_INDEX)
            child[i] = route
        
        # Remove empty routes
        child = [route for route in child if route]
        
        return child
        
    def mutate(self, solution, mutation_rate):
        """Mutação por troca (swap mutation) que respeita a estrutura de viagens."""
        if random.random() < mutation_rate:
            # Seleciona dois pontos para troca, garantindo que não sejam o depósito
            flat_solution = [gene for route in solution for gene in route]
            points_only = [gene for gene in flat_solution if gene != DEPOT_INDEX]

            if len(points_only) < 2:
                return solution # Não há pontos suficientes para mutação

            # Seleciona dois pontos de entrega para troca
            idx1, idx2 = sorted(random.sample(range(len(points_only)), 2))
            points_only[idx1], points_only[idx2] = points_only[idx2], points_only[idx1]

            # Reconstroi a rota com os pontos mutados
            new_solution_flat = []
            point_idx = 0
            for gene in flat_solution:
                if gene == DEPOT_INDEX:
                    new_solution_flat.append(DEPOT_INDEX)
                else:
                    new_solution_flat.append(points_only[point_idx])
                    point_idx += 1
            
            # Divide a rota mutada entre os veículos
            child = [[] for _ in range(MAX_VEHICLES)]
            current_idx = 0
            for i in range(MAX_VEHICLES):
                route_len = len([gene for gene in solution[i] if gene != DEPOT_INDEX]) + solution[i].count(DEPOT_INDEX)
                child[i] = new_solution_flat[current_idx:current_idx + route_len]
                current_idx += route_len
                
            return child

        return solution

    def run(self):
        """Executa o loop do algoritmo genético com otimizações de desempenho."""
        # Pré-calcula fitness para toda a população inicial
        fitness_cache = [self.calculate_fitness(sol) for sol in self.population]
        best_idx = min(range(len(fitness_cache)), key=lambda i: fitness_cache[i])
        best_solution = self.population[best_idx]
        best_cost = fitness_cache[best_idx]
        count_generations_without_improvement = 0
        mutation_rate = MUTATION_RATE

        for generation in range(GENERATIONS):
            new_population = []
            # Seleciona o melhor da geração atual
            best_gen_idx = min(range(len(fitness_cache)), key=lambda i: fitness_cache[i])
            best_of_gen = self.population[best_gen_idx]
            new_population.append(best_of_gen)
            new_fitness_cache = [fitness_cache[best_gen_idx]]

            if count_generations_without_improvement > 10:
                mutation_rate = min(0.5, mutation_rate * 1.2)
                count_generations_without_improvement = 0
                logging.info('Nova taxa de mutação: %s', mutation_rate)

            # Gera nova população usando seleção, crossover e mutação
            while len(new_population) < POPULATION_SIZE:
                parent1, parent2 = self.select_parents()
                child = self.crossover(parent1, parent2)
                mutated_child = self.mutate(child, mutation_rate)
                new_population.append(mutated_child)
                # Calcula fitness apenas para o novo indivíduo
                new_fitness_cache.append(self.calculate_fitness(mutated_child))

            self.population = new_population
            fitness_cache = new_fitness_cache
            current_best_idx = min(range(len(fitness_cache)), key=lambda i: fitness_cache[i])
            current_best = self.population[current_best_idx]
            current_best_cost = fitness_cache[current_best_idx]

            if current_best_cost < best_cost:
                best_solution = current_best
                best_cost = current_best_cost
                count_generations_without_improvement = 0
            else:
                count_generations_without_improvement += 1

            logging.info('Geração %d - Melhor Custo: %.2f', generation+1, best_cost)

        return best_solution, best_cost

# --- EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
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
                print(f"\nVeículo {i+1} não fez nenhuma entrega.")
                continue
                
            # Add depot at the start and end for display purposes only
            display_route = [DEPOT_INDEX] + route + [DEPOT_INDEX]
            
            total_route_duration = 0
            total_route_distance = 0
            
            last_point_idx = DEPOT_INDEX
            for point_idx in route:
                total_route_duration += duration_matrix[last_point_idx][point_idx]
                total_route_distance += distance_matrix[last_point_idx][point_idx]
                last_point_idx = point_idx
            
            # Add the return to the depot for the final total
            total_route_duration += duration_matrix[last_point_idx][DEPOT_INDEX]
            total_route_distance += distance_matrix[last_point_idx][DEPOT_INDEX]

            print(f"\nRota do Veículo {i+1}:")
            print(f"  Total de paradas: {len(route)}")
            print(f"  Tempo total de viagem: {total_route_duration / 60:.2f} minutos")
            print(f"  Distância total de viagem: {total_route_distance:.2f} metros")
            print(f"  Rota completa (índices): {display_route}")