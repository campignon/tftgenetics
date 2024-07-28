"""
Genetic Algorithm to generate viable TeamFight Tactics compositions.
"""
import random
import json
import numpy as np
from tqdm import tqdm

# Reference files
DATA_FILE = "TFTSet12_full_lookup.json"

# Algorithm parameters
POPULATION_SIZE = 200
ELITE_FRACTION = 0.15
TOURNAMENT_SIZE = 5
MUTATION_RATE = 0.05
GENERATIONS = 1000

NUM_ONES = 9  # Number of 1s for each individual (binary representation)
CHROMOSOME_LENGTH = 60

ELITE_SIZE = int(POPULATION_SIZE * ELITE_FRACTION)

# Export parameters
EXPORT_TOP_N = 100

def load_data(file):
    """Loads data from the data file"""
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def process_traits(traits_data):
    """Processes the traits data to extract and transform relevant data"""
    traits = []
    for trait in traits_data:
        units = [x["unit"] for x in trait["units"]]
        thresholds = [x["minUnits"] for x in trait["effects"]]
        traits.append({ "apiName": trait["apiName"], "units": units, "thresholds": thresholds })
    return traits

def create_traits_matrix(traits, units):
    """Creates the traits matrix from the traits data and units"""
    traits_matrix = np.zeros((len(units), len(traits)), dtype=int)
    for idx, trait in enumerate(traits):
        for unit in trait["units"]:
            unit_idx = units.index(unit)
            traits_matrix[unit_idx][idx] = 1
    return traits_matrix

def create_unique_individual(existing_individuals, length):
    """
    Creates a unique individual with exactly "num_ones" values equal to 1
    and the remaining equal to 0
    """
    while True:
        individual = np.zeros(length, dtype=int)
        ones_positions = random.sample(range(length), NUM_ONES)
        individual[ones_positions] = 1
        if not any(np.array_equal(individual, existing) for existing in existing_individuals):
            return individual

def create_population(size, length):
    """Creates a population of unique individuals"""
    population = []
    for _ in range(size):
        individual = create_unique_individual(population, length)
        population.append(individual)
    return np.array(population)

def compute_active_traits(individual, traits_matrix):
    """Compute the active traits for an individual/solution"""
    return np.sum(individual[:, np.newaxis] * traits_matrix, axis=0)

def compute_fitness(individual, traits_matrix, traits):
    """Computes the fitness of an individual/solution"""
    active_traits = compute_active_traits(individual, traits_matrix)
    fitness_value = 0
    for i, count in enumerate(active_traits):
        for threshold in traits[i]["thresholds"]:
            if count >= threshold:
                fitness_value += threshold
    return fitness_value

def tournament_selection(population, fitness_values, num_selected):
    """Tournament selection"""
    selected = []
    indices = list(range(len(population)))
    random.shuffle(indices)

    while len(selected) < num_selected:
        tournament_indices = random.sample(indices, TOURNAMENT_SIZE)
        tournament_individuals = population[tournament_indices]
        tournament_fitness = fitness_values[tournament_indices]
        winner_index = np.argmax(tournament_fitness)
        selected.append(tournament_individuals[winner_index])
        indices.remove(tournament_indices[winner_index])

    return np.array(selected)

def fix_child(child):
    """Makes sure the "child" argument is valid and fixes it otherwise"""
    # If there are too many elements whose value is equal to "1"
    if np.sum(child) > NUM_ONES:
        indices = np.where(child == 1)[0]
        to_remove = np.random.choice(indices, np.sum(child) - 8, replace=False)
        child[to_remove] = 0
    # If there are too few elements whose value is equal to "1"
    elif np.sum(child) < NUM_ONES:
        indices = np.where(child == 0)[0]
        to_add = np.random.choice(indices, 8 - np.sum(child), replace=False)
        child[to_add] = 1
    return child

def crossover(parent1, parent2):
    """Crossover between two parents to produce new individuals"""
    length = len(parent1)
    point = random.randint(1, length - 1)

    # Create two children by concatenating the two parents at the crossing point
    # and making sure they are valid
    child1 = fix_child(np.concatenate((parent1[:point], parent2[point:])))
    child2 = fix_child(np.concatenate((parent2[:point], parent1[point:])))

    return child1, child2

def mutate(individual):
    """Mutating by exchanging bits to ensure validity of the individuals"""
    if random.random() < MUTATION_RATE:
        # Find 1s and 0s
        ones_indices = np.where(individual == 1)[0]
        zeros_indices = np.where(individual == 0)[0]

        # Select a random 1 and a random 0
        one_idx = np.random.choice(ones_indices)
        zero_idx = np.random.choice(zeros_indices)

        # Exchange values
        individual[one_idx], individual[zero_idx] = individual[zero_idx], individual[one_idx]

    return individual

def compute_similarity(individual, existing_ind):
    """
    Compute the similarity between two individuals, counting only positions with at least one 1
    """
    # Identify positions with at least one 1 in either individual
    pos_with_one = np.where((individual | existing_ind) == 1)[0]

    return 2 - len(pos_with_one)/NUM_ONES

def is_diverse(individual, population, threshold=0.6):
    """Check if the individual is diverse enough compared to the population."""
    # sample_size = min(len(population), 100)  # Limit the sample size for diversity check
    # sampled_population = random.sample(list(population), sample_size)
    for existing_ind in population:
        similarity = compute_similarity(individual, existing_ind)
        if similarity > threshold:
            return False
    return True

def create_next_generation(num_children_to_create, parents, next_generation):
    """Create the next generation by combining elite with new offsprings"""
    children_created = 0
    while children_created < num_children_to_create:
        parent1, parent2 = random.sample(parents, 2)
        child1, child2 = crossover(parent1, parent2)
        if is_diverse(child1, next_generation) and is_diverse(child2, next_generation):
            next_generation.extend([child1, child2])
            children_created += 2

    next_generation = next_generation[:POPULATION_SIZE]

    return np.array(next_generation)

def display_solution(individual, units, fitness):
    """Display solution with units names"""
    display_units = [units[i] for i in range(len(individual)) if individual[i] == 1]
    return f"score = {fitness}, units : {display_units}"

def generate_compositions(traits, traits_matrix):
    """Main function which runs the genetic algorithm"""
    # Creating the initial population
    population = create_population(POPULATION_SIZE, CHROMOSOME_LENGTH)

    with tqdm(total=GENERATIONS, desc="Generations") as pbar:
        for generation in range(GENERATIONS):
            # Evaluating population
            fitness_values = np.array([
                compute_fitness(ind, traits_matrix, traits) for ind in population
            ])

            # Sorting population by decreasing fitness
            sorted_indices = np.argsort(fitness_values)[::-1]
            population = population[sorted_indices]
            fitness_values = fitness_values[sorted_indices]

            # Updating max fitness
            pbar.set_postfix({"Max Fitness": fitness_values[0]})

            # If we have reached the last generation
            if generation == GENERATIONS - 1:
                break

            # Elite selection
            next_generation = population[:ELITE_SIZE].tolist()

            # Tournament selection
            num_selected = POPULATION_SIZE - ELITE_SIZE
            parents = tournament_selection(population, fitness_values, num_selected)

            # Creating new offsprings
            num_offsprings_to_create = POPULATION_SIZE - len(next_generation)
            next_generation = create_next_generation(
                num_offsprings_to_create,
                list(parents),
                next_generation
            )

            # Mutation
            for individual in next_generation[ELITE_SIZE:]:
                mutate(individual)

            # Replacing population with next_generation
            population = next_generation

            # Updating progress bar
            pbar.update(1)

    return population, fitness_values

def export_top_solutions(population, fitness_values, units):
    """Export the top N solutions to a JSON file"""
    top_solutions = []
    for idx, individual in enumerate(population[:EXPORT_TOP_N]):
        solution = {
            "score": int(fitness_values[idx]),
            "units": [units[i] for i in range(len(individual)) if individual[i] == 1] 
        }
        top_solutions.append(solution)

    with open("top_solutions.json", "w", encoding="utf-8") as f:
        json.dump(top_solutions, f, ensure_ascii=False, indent=4)

def main():
    """Entry point of the script"""
    # Loading and processing data
    data = load_data(DATA_FILE)
    units = [x["apiName"] for x in data["units"] if x["traits"]]
    traits = process_traits(data["traits"])
    traits_matrix = create_traits_matrix(traits, units)
    # Running the algorithm
    population, fitness_values = generate_compositions(traits, traits_matrix)
    # Exporting the N best solutions
    export_top_solutions(population, fitness_values, units)

if __name__ == "__main__":
    main()
