"""
Configuration constants and parameters for the VRP Genetic Algorithm.
This module centralizes all configuration parameters and constants used throughout the system.
"""

# Algorithm parameters
TWO_OPT_FREQUENCY = 10  # Apply local search every X generations
COUNT_GENERATIONS_WITHOUT_IMPROVEMENT = 50
COUNT_GENERATIONS_WITHOUT_IMPROVEMENT_FOR_MUTATION = 5
DEFAULT_MUTATION_RATE = 0.05

# Time constants (in seconds)
TIME_DEPOT_STOP = 180  # 3 minutes per depot stop

# VRP constants
DEPOT_INDEX = 0

# Selection parameters
TOURNAMENT_SIZE = 5

# Weights
DEFAULT_TIME_WEIGHT = 0.5
DEFAULT_DISTANCE_WEIGHT = 0.5

# Mutation parameters
MUTATION_RATE_INCREASE_FACTOR = 1.1
MAX_MUTATION_RATE = 0.5

# Crossover probabilities
TWO_OPT_MUTATION_PROBABILITY = 0.5

# Local search parameters
MAX_IMPROVEMENTS_2OPT = 1
MAX_ITER_WITHOUT_IMPROVE_INTER_ROUTE = 1

# Penalty factors
DURATION_PENALTY_THRESHOLD = 0.9
DISTANCE_PENALTY_THRESHOLD = 0.9
PENALTY_MULTIPLIER = 5.0
