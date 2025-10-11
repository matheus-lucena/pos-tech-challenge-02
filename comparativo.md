# Overview projeto
O intuito deste projeto foi fornecer um VRP para otimização de rotas para entregas de insumos médicos, sendo que cada ponto atribuido no mapa representa uma entrega. Neste desenvolvimento foi utilizado Algoritmo Genético + Integração com o OSRM para nos prover distâncias e tempo para percorrer o trajeto com base em um cenário real (com base em dados de mapa). O projeto inclui uma interface WEB que permite visualizar e configurar parametros, sendo possível acompanhar em tempo real as optimizacoes.

### Principais funcionalidades
* Algoritmo Genético híbrido com operadores específicos para VRP (crossover, mutation, selection).
* Local search: 2-opt, inter-route relocate/swap/balance.
* Integração com OSRM para matrizes reais de distância e tempo.
* Visualização em tempo real (Leaflet) com painel de status de veículos.
* Integração com LLM (ex.: Gemini) para geração automática de relatórios e instruções.
* Execução reproducível via docker-compose (inclui serviço OSRM).

### Requisitos para execuçao

* Python 3.10+
* Docker & Docker Compose (para OSRM e infra local)
* (Opcional) Conta/API key do provedor LLM (definir em .env)

# Comparativo técnico com outras soluções VRP do mercado:
Comparativo técnico com outras soluções VRP do mercado
Soluções comparadas
* Google OR-Tools (Routing Solver)
* VROOM (backend para roteirização)

### Resumo características por produto
#### Google OR-Tools
Pontos fortes:
* Performace altissima.
* Suporte a capacidades, janelas de tempo, custos variados.
* Excelente documentação e comunidade.
* Limitações:
  * Menos “interpretável” quando se usa CP-SAT; difícil customizar heurísticas específicas (embora possível).
  * Integração com LLMs/reports não nativa — precisa camada adicional.

#### VROOM
Pontos fortes:
* Fácil de integrar (baseado em API Rest JSON).
* Alta razao em otimizacoes
* Limitações:
  * Customizacao mais limitada
  * Uso ideal: Sistema que precisa de serviço de roteirização rápido sem desenvolver solver.

# Tabela comparativa

| **Critério / Sistema**              | **OR-Tools**     | **VROOM**       | **Seu Sistema**                            |
|------------------------------------|------------------|-----------------|---------------------------------------------|
| **Linguagem**                      | C++ / Python     | C++ / API       | Python                                      |
| **Customização de operadores**     | Média / Alta (via API) | Baixa     | 🚀 **Muito Alta**                          |
| **Suporte a janelas de tempo**     | ✅ Sim           | ✅ Sim          | ❌ Não  (Porém possível)                     |
| **Integração LLM / relatórios**    | ❌ Não nativo     | ❌ Não          | ✅ **Nativo (LLM + PDF)**                  |
| **Facilidade de implantação**      | ✅ Alta          | ✅ Alta         | ✅ **Alta (docker-compose)**               |
| **Performance em grande escala**   | 🚀 Excelente     | 🚀 Excelente    | ⚙️ Ok (precisa de otimizacoes e benchmarks)   |
| **Transparência dos passos**       | ⚙️ Baixa / Média | ⚙️ Baixa        | ✅ **Alta (código aberto)**                |
| **Custos de licença**                         | 💸 Livre         | 💸 Livre        | 💸 **Livre**                              |


## Vantagens sistema desenvolvido

* Integração com LLMs
* Visualização em tempo real

## Desvantagens / limitações

* Performance e robustez em escala — soluções como OR-Tools/VROOM podem ser mais rápidas e otimizadas em casos muito grandes (centenas/milhares de pontos).
* Validação de mercado
* Complexidade de otimizacao — Algoritmo genetico requer monitoramento de parametros (mutation rate, population size)


### Exemplo implementaçao propria GA
```python
    import math
    import random

    from vrp.vrp_ga import VRPGeneticAlgorithm

    num_vehicles = 5
    vehicle_autonomy = 100_000

    def euclidean_distance(p1, p2):
        dx = (p1[0] - p2[0]) * 111000 
        dy = (p1[1] - p2[1]) * 111000
        return int(math.sqrt(dx * dx + dy * dy))

    random.seed(42)
    points = [(-22.421, -47.580)]
    for _ in range(29):
        lat = -22.421 + random.uniform(-0.02, 0.02)
        lon = -47.580 + random.uniform(-0.02, 0.02)
        points.append((lat, lon))

    distance_matrix = [
        [euclidean_distance(points[i], points[j]) for j in range(len(points))]
        for i in range(len(points))
    ]

    time_matrix = [
        [random.randint(200, 4000) if i != j else 0 for j in range(len(points))]
        for i in range(len(points))
    ]

    ga = VRPGeneticAlgorithm(
        time_matrix,
        distance_matrix,
        points=points,
        max_vehicles=num_vehicles,
        vehicle_max_points=8,
        generations=25,
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
            acc_time += time_matrix[from_idx][to_idx]
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
```

### Exemplo ortools

#### Requisitos
```
pip install ortools
```

#### Código
```python
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math
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

def euclidean_distance(p1, p2):
    dx = (p1[0] - p2[0]) * 111000 
    dy = (p1[1] - p2[1]) * 111000
    return int(math.sqrt(dx * dx + dy * dy))

distance_matrix = [
    [euclidean_distance(points[i], points[j]) for j in range(len(points))]
    for i in range(len(points))
]

time_matrix = [
    [random.randint(200, 4000) if i != j else 0 for j in range(len(points))]
    for i in range(len(points))
]

demands = [0] + [1 for _ in range(len(points) - 1)]

manager = pywrapcp.RoutingIndexManager(len(points), num_vehicles, depot_index)
routing = pywrapcp.RoutingModel(manager)

def distance_time_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    distance = distance_matrix[from_node][to_node]
    time = time_matrix[from_node][to_node]

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
                acc_time += time_matrix[prev_node][node_index]
            stop_times.append(acc_time)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        route.append(depot_index)
        # Última parada (retorno ao depósito)
        if len(route) > 1:
            acc_time += time_matrix[route[-2]][route[-1]]
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
```
