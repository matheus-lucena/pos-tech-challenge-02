import requests
import logging
import json
from vrp.llmintegration import gerar_pdf_relatorio
from vrp.vrp_ga import VRPGeneticAlgorithm

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

# --- PARÂMETROS OTIMIZADOS PARA PERFORMANCE ---
OSRM_BASE_URL = "http://localhost:5001/table/v1/driving/"
POPULATION_SIZE = 100          # ⚡ Reduzido de 400 para 100 (4x mais rápido)
POPULATION_HEURISTIC_TAX = 0.7 # ⚡ Aumentado para melhor população inicial
GENERATIONS = 20
MUTATION_RATE = 0.08           # ⚡ Aumentado para compensar menor população
MAX_VEHICLES = 20
VEHICLE_MAX_POINTS = 8  # Max number of stops per vehicle
MAX_TRIP_DURATION = 8 * 3600  # X hours in seconds
MAX_TRIP_DISTANCE = 5000000  # Max distance in meters
TIME_TO_STOP = 180  # 3 minutes in seconds per stop
TIME_DEPOT_STOP = 180  # 3 minutes in seconds per stop
TWO_OPT_FREQUENCY = 100  # Apply local search every X generations

COUNT_GENERATIONS_WITHOUT_IMPROVEMENT = 25  # ⚡ Reduzido para restart mais rápido
COUNT_GENERATIONS_WITHOUT_IMPROVEMENT_FOR_MUTATION = 5

DEPOT_INDEX = 0  # Assuming the first point is the depot


def generate_json_output(best_solution, best_cost, duration_matrix, distance_matrix, points, wait_time, num_vehicles, vehicle_max_points, max_trip_duration):
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

            current_trip_duration += TIME_DEPOT_STOP  # Tempo inicial no depósito

            for point_idx in travel[1:]:
                # Se for um ponto intermediário de viagem/rota
                if last_point_idx != -1 and point_idx != -1:
                    current_trip_duration += duration_matrix[last_point_idx][point_idx]
                    current_trip_distance += distance_matrix[last_point_idx][point_idx]

                # Tempo de parada
                if point_idx != DEPOT_INDEX:
                    current_trip_duration += wait_time
                else:
                    current_trip_duration += TIME_DEPOT_STOP

                last_point_idx = point_idx

            total_route_duration += current_trip_duration
            total_route_distance += current_trip_distance

            formatted_travel_points = "/".join(
                [f"{points[idx][0]},{points[idx][1]}" for idx in travel])

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
            "full_route_coordinates": "/".join([f"{points[idx][0]},{points[idx][1]}" for idx in route]),
            "trips": travels_data
        })

    final_output = {
        "best_normalized_cost": best_cost,
        "number_of_vehicles_used": len(all_routes_data),
        "cost_matrix_based_on": OSRM_BASE_URL,
        "base_constraints": {
            "max_fleet_vehicles": num_vehicles,
            "max_stops_per_trip": vehicle_max_points,
            "max_duration_minutes": max_trip_duration / 60
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
        logging.error(
            "Erro na resposta do OSRM: 'durations/distances' não encontrado.")
        return None, None


def run_vrp(points: list, max_epochs: int, num_vehicles: int, vehicle_max_points: int, max_trip_distance: int, max_trip_duration: int, wait_time: int = 3, mutation_rate: float = 0.05, max_no_improvement: int = 50, epoch_callback: callable = None, generate_json: bool = False):
    duration_matrix, distance_matrix = get_cost_matrix(points)
    
    if duration_matrix and distance_matrix:
        print("Matriz de distâncias recebida com sucesso.")

        print(
            f"\n2. Executando Algoritmo Genético com {num_vehicles} veículos...")
        ga = VRPGeneticAlgorithm(
            duration_matrix,
            distance_matrix,
            points=points,
            max_vehicles=num_vehicles,
            vehicle_max_points=vehicle_max_points,
            generations=max_epochs,
            max_trip_distance=max_trip_distance,
            max_trip_duration=max_trip_duration,
            time_to_stop=wait_time,
            mutation_rate=mutation_rate,
            max_no_improvement=max_no_improvement,
            population_size=POPULATION_SIZE,
            population_heuristic_tax=POPULATION_HEURISTIC_TAX,
        )
        best_solution, best_cost = ga.run(epoch_callback)

        if generate_json:
            print("\n--- INÍCIO DA SAÍDA JSON PARA LLM ---")
            final_json_data = generate_json_output(
                best_solution,
                best_cost,
                duration_matrix,
                distance_matrix,
                points,
                wait_time,
                num_vehicles,
                vehicle_max_points,
                max_trip_duration
            )

            json_output = json.dumps(final_json_data, indent=4, ensure_ascii=False)

            print(json_output)
            print("--- FIM DA SAÍDA JSON PARA LLM ---")
            print("\n--- Resultado Final ---")

        print(
            f"Melhor custo de solução encontrado: {best_cost:.2f} (em unidades normalizadas)")
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
