import requests
import logging
from vrp.llmintegration import generate_pdf_report
from vrp.vrp_ga import VRPGeneticAlgorithm
from vrp.config import DEPOT_INDEX, TIME_DEPOT_STOP, OSRM_BASE_URL, POPULATION_SIZE, POPULATION_HEURISTIC_TAX

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

def generate_json_output(best_solution, best_cost, duration_matrix, distance_matrix, points, wait_time, num_vehicles, vehicle_max_points, max_trip_duration):
    all_routes_data = []

    for i, route in enumerate(best_solution):
        if not route:
            continue

        travels_data = []
        travels = []
        current_travel = []

        for point_idx in route:
            current_travel.append(point_idx)
            if point_idx == DEPOT_INDEX and len(current_travel) > 1:
                travels.append(current_travel)
                current_travel = [DEPOT_INDEX]

        if len(current_travel) > 1:
            travels.append(current_travel)

        total_route_duration = 0
        total_route_distance = 0

        for j, travel in enumerate(travels):

            current_trip_duration = 0
            current_trip_distance = 0
            last_point_idx = travel[0]

            current_trip_duration += TIME_DEPOT_STOP 

            for point_idx in travel[1:]:
                if last_point_idx != -1 and point_idx != -1:
                    current_trip_duration += duration_matrix[last_point_idx][point_idx]
                    current_trip_distance += distance_matrix[last_point_idx][point_idx]

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

    generate_pdf_report(final_output) 
    return final_output


def get_matrix(locations):
    coords_str = ";".join([f"{lon},{lat}" for lat, lon in locations])
    url = f"{OSRM_BASE_URL}{coords_str}?sources=all&destinations=all&annotations=duration,distance"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error("Error to connect ao OSRM: %s", e)
        return None


def get_cost_matrix(locations):
    data = get_matrix(locations)
    if data and 'durations' in data and 'distances' in data:
        return data['durations'], data['distances']
    else:
        logging.error(
            "Error: 'durations/distances' not found")
        return None, None


def run_vrp(points: list, max_epochs: int, num_vehicles: int, vehicle_max_points: int, max_trip_distance: int, max_trip_duration: int, wait_time: int = 3, mutation_rate: float = 0.05, max_no_improvement: int = 50, epoch_callback: callable = None, generate_json: bool = False):
    duration_matrix, distance_matrix = get_cost_matrix(points)
    
    if duration_matrix and distance_matrix:
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
            generate_json_output(
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

        for _, route in enumerate(best_solution):
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
