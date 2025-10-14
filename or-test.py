from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import requests
import random

random.seed(42)
points = [(-22.421, -47.580)]
for _ in range(29):
    lat = -22.421 + random.uniform(-0.02, 0.02)
    lon = -47.580 + random.uniform(-0.02, 0.02)
    points.append((lat, lon))

num_vehicles = 5
vehicle_capacities = [8] * num_vehicles
vehicle_autonomy = 100_000
depot_index = 0

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


demands = [0] + [1 for _ in range(len(points) - 1)]

manager = pywrapcp.RoutingIndexManager(len(points), num_vehicles, depot_index)
routing = pywrapcp.RoutingModel(manager)

def distance_time_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    distance = distance_matrix[from_node][to_node]
    time = duration_matrix[from_node][to_node]

    combined_cost = 0.5 * distance + 0.5 * time
    return int(combined_cost)

transit_callback_index = routing.RegisterTransitCallback(distance_time_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

def demand_callback(from_index):
    from_node = manager.IndexToNode(from_index)
    return demands[from_node]

demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
routing.AddDimensionWithVehicleCapacity(
    demand_callback_index,
    0,
    vehicle_capacities,  
    True, 
    'Capacity'
)

routing.AddDimension(
    transit_callback_index,
    0,
    vehicle_autonomy,
    True,
    'DistanceTime'
)

# Parâmetros de busca
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
search_parameters.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
search_parameters.time_limit.FromSeconds(5)

solution = routing.SolveWithParameters(search_parameters)

if solution:
    print("Solução encontrada!")
    total_distance = 0
    total_time = 0
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route = []
        route_distance = 0
        stop_times = []
        acc_time = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(node_index)
            if len(route) > 1:
                prev_node = route[-2]
                acc_time += duration_matrix[prev_node][node_index]
            stop_times.append(acc_time)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        route.append(depot_index)
        # Última parada (retorno ao depósito)
        if len(route) > 1:
            acc_time += duration_matrix[route[-2]][route[-1]]
        stop_times.append(acc_time)
        total_distance += route_distance
        # Calcula o tempo total da rota
        route_time = acc_time
        if len(route) > 2:
            print(f"Veículo {vehicle_id+1}: {route} | Distância: {route_distance/1000:.2f} km | Tempo: {route_time/60:.2f} min")
            for idx, (node, t) in enumerate(zip(route, stop_times)):
                print(f"  Parada {idx}: Ponto {node}, Tempo acumulado: {t/60:.2f} min")
        total_time += route_time

    print(f"Distância total: {total_distance/1000:.2f} km")
    print(f"Tempo total: {total_time/60:.2f} min")