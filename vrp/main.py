import requests
import logging
import time
import json
from vrp.points import POINTS
from vrp.llmintegration import gerar_pdf_relatorio
from vrp.genetic_algorithm import VRPGeneticAlgorithm

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

# --- PARÂMETROS ---
OSRM_BASE_URL = "http://localhost:5001/table/v1/driving/"
POPULATION_SIZE = 400
POPULATION_HEURISTIC_TAX = 0.5
GENERATIONS = 20
MUTATION_RATE = 0.05
MAX_VEHICLES = 20
VEHICLE_MAX_POINTS = 8  # Max number of stops per vehicle
MAX_TRIP_DURATION = 8 * 3600  # X hours in seconds
MAX_TRIP_DISTANCE = 50000  # Max distance in meters
TIME_TO_STOP = 180  # 3 minutes in seconds per stop
TIME_DEPOT_STOP = 180  # 3 minutes in seconds per stop
TWO_OPT_FREQUENCY = 100  # Apply local search every X generations

COUNT_GENERATIONS_WITHOUT_IMPROVEMENT = 50
COUNT_GENERATIONS_WITHOUT_IMPROVEMENT_FOR_MUTATION = 5

DEPOT_INDEX = 0  # Assuming the first point is the depot


def generate_json_output(best_solution, best_cost, duration_matrix, distance_matrix, points):
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
                    current_trip_duration += TIME_TO_STOP
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
        logging.error(
            "Erro na resposta do OSRM: 'durations/distances' não encontrado.")
        return None, None


def run_vrp(points: list, max_epochs: int, num_vehicles: int, epoch_callback: callable = None):
    duration_matrix, distance_matrix = get_cost_matrix(POINTS)

    if duration_matrix and distance_matrix:
        print("Matriz de distâncias recebida com sucesso.")

        print(
            f"\n2. Executando Algoritmo Genético com {MAX_VEHICLES} veículos...")
        ga = VRPGeneticAlgorithm(
            duration_matrix,
            distance_matrix,
            points=points,
            max_vehicles=num_vehicles,
            vehicle_max_points=VEHICLE_MAX_POINTS,
            generations=max_epochs,
            population_size=POPULATION_SIZE,
            population_heuristic_tax=POPULATION_HEURISTIC_TAX,
            max_trip_duration=MAX_TRIP_DURATION,
            max_trip_distance=MAX_TRIP_DISTANCE,
        )
        best_solution, best_cost = ga.run(epoch_callback)

        final_json_data = generate_json_output(
            best_solution,
            best_cost,
            duration_matrix,
            distance_matrix,
            points
        )

        json_output = json.dumps(final_json_data, indent=4, ensure_ascii=False)

        print("\n--- INÍCIO DA SAÍDA JSON PARA LLM ---")
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

            print(f"\nRota do Veículo {i+1}:")

            for j, travel in enumerate(travels):

                trip_cost, total_travel_duration, total_travel_distance, stops = ga._get_trip_cost(
                    travel)

                if total_travel_duration == 0:
                    continue

                print(f"     Viagem {j+1}:")
                print(f"       Paradas: {travel[1:-1]}")
                print(f"       Total de paradas de entrega: {stops}")
                print(
                    f"       Tempo de viagem: {total_travel_duration / 60:.2f} minutos")
                print(
                    f"       Distância de viagem: {total_travel_distance:.2f} metros")
                print(f"       Rota (índices): {travel}")

                formatted_travel_points = "/".join(
                    [f"{POINTS[idx][0]},{POINTS[idx][1]}" for idx in travel])
                print(
                    f"       String formatada das paradas da viagem: {formatted_travel_points}")

            formatted_route_points = "/".join(
                [f"{POINTS[idx][0]},{POINTS[idx][1]}" for idx in route])
            print(
                f"   String formatada das paradas da rota do veículo: {formatted_route_points}")


# --- EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    start_time = time.time()

    run_vrp(
        points=POINTS,
        max_epochs=GENERATIONS,
        num_vehicles=MAX_VEHICLES
    )

    end_time = time.time()
    logging.info("Tempo total de execução: %.2f segundos",
                 end_time - start_time)

    print(POINTS[0])
