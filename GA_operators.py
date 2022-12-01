import numpy as np
from deap import creator, tools

"""
File that contains all the operators of the GA: initialization, selection, mating, mutation, clonation and selection
"""

def initialize_individual(individual_class, parameters):

    """
    Function that initializes an individual randomly

    :param individual_class: class to which individuals belong to
    :param parameters: parameters of the algorithm

    :return: initialized individual
    """

    genotype_length = np.random.randint(parameters.min_initial_len, parameters.max_initial_len)
    return individual_class(np.random.randint(0, 255, genotype_length))


def parent_selection(population, parameters, toolbox):

    """
    Parent selection operator. Tournament selection

    :param population: current population
    :param parameters: parameters of the algorithm

    :return: selected parents
    """
    return tools.selTournament(population, k=parameters.population_length, tournsize=parameters.tournament_size)


def mating(parameters, parents, toolbox):

    """
    Mating operator. One point mating but using different point on each parent so as to obtain sons of
    variable length

    :param parameters: parameters of the algorithm
    :param parents: selected parents

    :return: mated offspring
    """

    offspring = []
    cross_number = len(parents) // 2
    current_parent_idx = 0

    for _ in range(cross_number):

        parent1 = parents[current_parent_idx]
        parent2 = parents[current_parent_idx + 1]

        if np.random.random() < parameters.mating_probability:

            point1 = np.random.randint(0, len(parent1)-1)
            point2 = np.random.randint(0, len(parent2)-1)

            son1 = creator.Individual(parent1[:point1] + parent2[point2:])
            son2 = creator.Individual(parent1[point1:] + parent2[:point2])

        else:

            son1 = toolbox.clone(parent1)
            son2 = toolbox.clone(parent2)

        offspring.append(son1)
        offspring.append(son2)

        current_parent_idx += 2

    return offspring


def mutation(offspring, parameters):

    """
    Mutation operator. Mutation gen by gen with a pm probability of mutation

    :param offspring: mated offspring
    :param parameters: parameters of the algorithm

    :return: mutated offspring
    """

    for mutant in offspring:
        tools.mutUniformInt(mutant, low=0, up=255, indpb=parameters.mutation_prob)

    return offspring


def duplication(offspring, parameters, toolbox):

    """
    Duplication operator. Appends randomly a subchain (with random length between its limits) of each genotype
    at its end with a pd probability of duplication

    :param offspring: mutated offspring
    :param parameters: parameters of the algorithm

    :return: duplicated offspring
    """

    duplicated_individuals = []

    for individual in offspring:
        if np.random.random() < parameters.duplication_probability:
            number_of_codons_to_duplicate = np.random.randint(parameters.min_duplication_len, max(len(individual),
                                                                                                  parameters.max_duplication_len))

            duplication_chain_starting_point = np.random.randint(0, len(individual)-1)

            duplicated_individual = creator.Individual(individual + individual[duplication_chain_starting_point:\
                                                                               duplication_chain_starting_point +\
                                                                               number_of_codons_to_duplicate])

        else:
            duplicated_individual = toolbox.clone(individual)

        duplicated_individuals.append(duplicated_individual)

    return duplicated_individuals


def survival_selection(population, offspring, parameters, toolbox):

    """
    Survival selection operator. Generational model with the posibility of elitism, mu + lambda or a mixture of both

    :param population: current population
    :param offspring: current offspring
    :param parameters: parameters of the algorithm

    :return: selected new population
    """
    if parameters.survival_selection_method == "generational":

        if parameters.elitism:
            population_fitnesses = [individual.fitness.values[0] for individual in population]
            offspring_fitnesses = [individual.fitness.values[0] for individual in offspring]

            if min(population_fitnesses) < min(offspring_fitnesses):

                best_individual_idx = np.argmin(population_fitnesses)
                random_individual_idx = np.random.randint(0, len(offspring))

                offspring[random_individual_idx] = toolbox.clone(population[best_individual_idx])

        return offspring

    elif parameters.survival_selection_method == "mu_plus_lambda":

        return tools.selBest(population + offspring, k=parameters.population_length)

    elif parameters.survival_selection_method == "steady_state_model":

        best_ind_pop = tools.selBest(population, k=parameters.population_length // 2)
        best_ind_off = tools.selBest(offspring, k=parameters.population_length // 2)

        return best_ind_pop + best_ind_off