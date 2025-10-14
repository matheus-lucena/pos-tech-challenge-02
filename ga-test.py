import math
import random
import requests

from vrp.vrp_ga import VRPGeneticAlgorithm

num_vehicles = 5
vehicle_autonomy = 100_000

random.seed(42)
points = [(-22.421, -47.580)]
for _ in range(29):
    lat = -22.421 + random.uniform(-0.02, 0.02)
    lon = -47.580 + random.uniform(-0.02, 0.02)
    points.append((lat, lon))

def get_matrix(locations):
    coords_str = ";".join([f"{lon},{lat}" for lat, lon in locations])
    url = f"http://localhost:5001/table/v1/driving/{coords_str}?sources=all&destinations=all&annotations=duration,distance"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return None


def get_cost_matrix(locations):
    data = get_matrix(locations)
    if data and 'durations' in data and 'distances' in data:
        return data['durations'], data['distances']
    else:
        return None, None

duration_matrix, distance_matrix = get_cost_matrix(points)


ga = VRPGeneticAlgorithm(
    duration_matrix,
    distance_matrix,
    points=points,
    max_vehicles=num_vehicles,
    vehicle_max_points=8,
    generations=150,
    max_trip_distance=vehicle_autonomy,
    max_trip_duration=8 * 60 * 60,
    time_to_stop=180,
    mutation_rate=0.15,
    max_no_improvement=2,
    population_size=75,
    population_heuristic_tax=0.7,
)
best_solution, best_cost = ga.run()

print("Solução encontrada!")
total_distance = 0
total_time = 0
for vehicle_id, route in enumerate(best_solution):
    if not route:
        continue
    route_distance = 0
    stop_times = []
    acc_time = 0
    for idx in range(len(route) - 1):
        from_idx = route[idx]
        to_idx = route[idx + 1]
        route_distance += distance_matrix[from_idx][to_idx]
        acc_time += duration_matrix[from_idx][to_idx]
        stop_times.append(acc_time)
    total_distance += route_distance
    route_time = acc_time
    if len(route) > 1:
        print(f"Veículo {vehicle_id+1}: {route} | Distância: {route_distance/1000:.2f} km | Tempo: {route_time/60:.2f} min")
        for idx, (node, t) in enumerate(zip(route[1:], stop_times)):
            print(f"  Parada {idx+1}: Ponto {node}, Tempo acumulado: {t/60:.2f} min")
    total_time += route_time
print(f"Distância total: {total_distance/1000:.2f} km")
print(f"Tempo total: {total_time/60:.2f} min")