import numpy as np
from deap import base, creator, tools
from parameters import Parameters
from GA_operators import *
from  GE_fitness import evaluate_individual
from decode_backus_naur import decode_individual
import statistics
import math
from restrictions import manage_adaptative_restrictions
from local_search import local_search

"""
File that contains operators corresponding to the GE (creation and launching)
"""


def run_GE(parameters, verbose=True):

    """
    Function that runs the GE

    :param parameters: parameters of the algorithm
    :param verbose: whether to print logs or not

    :return: execution of GA_algorithm (see below)
    """

    toolbox = create_GA_classes(parameters)
    toolbox = setup_algorithm(parameters, toolbox)

    return GA_algorithm(parameters, toolbox, verbose=verbose)


def create_GA_classes(parameters):

    """


    :param parameters:
    :return:
    """

    creator.create("Fitness", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.Fitness)

    toolbox = base.Toolbox()
    toolbox.register("individual", initialize_individual, creator.Individual, parameters)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    return toolbox


def setup_algorithm(parameters, toolbox):

    """
    Function that setups each operator of the GA

    :param parameters: parmeters of the algorithm
    :param toolbox: toolbox of the algorithm

    :return:
    """

    toolbox.register("select", parent_selection)
    toolbox.register("mate", mating)
    toolbox.register("mutate", mutation)
    toolbox.register("duplicate", duplication)
    toolbox.register("survival_selection", survival_selection)

    toolbox.register("evaluate", evaluate_individual, parameters=parameters)

    return toolbox


def log_fitness_and_length(parameters, population, fitnesses, generation):

    """
    Logs fitness and length parameters on each genration

    :param parameters: parameters of the algorithm
    :param population: current population
    :param fitnesses: current fitness values
    :param generation: current generation

    :return:
    """

    fitnesses = [value for value in fitnesses if value < parameters.fitness_for_invalid_individuals]

    mean_fitness = sum(fitnesses) / len(fitnesses)
    min_fitness = min(fitnesses)
    sd_fitnesses = statistics.stdev(fitnesses)

    parameters.avg_fitnesses[generation] = mean_fitness
    parameters.min_fitnesses[generation] = min_fitness
    parameters.sd_fitnesses[generation] = sd_fitnesses

    genotype_lengths = [len(ind) for ind in population]
    min_len = min(genotype_lengths)
    max_len = max(genotype_lengths)
    avg_len = sum(genotype_lengths) / len(genotype_lengths)

    parameters.min_lens[generation] = min_len
    parameters.max_lens[generation] = max_len
    parameters.avg_lens[generation] = avg_len


def log_wrapping(parameters, generation):
    """
    Logs current wrapping values

    :param parameters: algorithm parameters
    :param generation: current generation

    :return:
    """

    mean_wrapping = sum(parameters.wrappings_by_individual) / len(parameters.wrappings_by_individual)
    std_wrapping = statistics.stdev(parameters.wrappings_by_individual)

    parameters.avg_wrapping[generation] = mean_wrapping
    parameters.sd_wrapping[generation] = std_wrapping

    parameters.wrappings_by_individual = []


def GA_algorithm(parameters, toolbox, verbose=True):

    """
    Implementation of the GA

    :param parameters: parameters of the algorithm
    :param toolbox: toolbox of the algorithm
    :param verbose: whether to print logs

    :return: results of the executions
    """

    # Initialization of the population
    parameters.clear_logs()
    population = toolbox.population(parameters.population_length)

    number_of_evaluations = 0

    # Evaluation
    fitnesses = [toolbox.evaluate(individual) for individual in population]
    fitnesses_list = []
    for individual, fitness in zip(population, fitnesses):
        if math.isnan(fitness[0]):
            fitness = (parameters.fitness_for_invalid_individuals, False)
        individual.fitness.values = fitness
        fitnesses_list.append(fitness[0])
        number_of_evaluations += 1

    success = False

    generation = 0

    log_fitness_and_length(parameters, population, fitnesses_list, generation)
    log_wrapping(parameters, generation)

    if verbose:
        print("generation\tmin fitness\tavg fitness\tstd fitness")
        print("-------------------------------------------------------------")
        print(f"{generation}\t{parameters.min_fitnesses[generation]}\t{parameters.avg_fitnesses[generation]}\t{parameters.sd_fitnesses[generation]}")

    while generation < parameters.max_gens:

        generation += 1
        parameters.compute_mutation_prob(generation)
        parameters.compute_tournament_size(generation)

        # Selection
        parents = toolbox.select(population, parameters, toolbox)

        # Mating
        offspring = toolbox.mate(parameters, parents, toolbox)

        # Mutation
        mutated_offspring = toolbox.mutate(offspring, parameters)

        # Duplication
        duplicated_offspring = toolbox.duplicate(mutated_offspring, parameters, toolbox)

        # Evaluation
        offspring_fitnesses = [toolbox.evaluate(individual) for individual in duplicated_offspring]
        offspring_fitnesses_list = []
        for ind, fit in zip(duplicated_offspring, offspring_fitnesses):
            if math.isnan(fit[0]):
                fit = parameters.fitness_for_invalid_individuals, False
            ind.fitness.values = fit
            offspring_fitnesses_list.append(fit[0])
            number_of_evaluations += 1

        # Local search
        if generation % 10 == 0:
            for idx, ind in enumerate(duplicated_offspring):
                duplicated_offspring[idx], evaluations = local_search(ind, parameters, toolbox)
                number_of_evaluations += evaluations

        # Survival selection
        population = survival_selection(population, duplicated_offspring, parameters, toolbox)

        assert len(population) == parameters.population_length

        fitnesses = [toolbox.evaluate(individual) for individual in population]
        fitnesses_list = [ind.fitness.values[0] for ind in population]

        best_individual_idx = np.argmin(fitnesses_list)
        best_individual = population[best_individual_idx]
        best_individuals_fitness_value = fitnesses[best_individual_idx]

        # For restrictions
        if best_individuals_fitness_value[1]:
            parameters.best_individual_factible_integrationConst += 1

        if len(best_individual) <= parameters.max_genotype_len:
            parameters.best_individual_factible_times_populationLen += 1

        recalculate = False
        if generation % parameters.Nf == 0:
            recalculate = manage_adaptative_restrictions(parameters)

        if recalculate:
            fitnesses_list = []
            fitnesses = [toolbox.evaluate(individual) for individual in population]
            for individual, fitness in zip(population, fitnesses):
                if math.isnan(fitness[0]):
                    fitness = (parameters.fitness_for_invalid_individuals, False)
                individual.fitness.values = fitness
                fitnesses_list.append(fitness[0])
                number_of_evaluations += 1

        log_fitness_and_length(parameters, population, fitnesses_list, generation)
        log_wrapping(parameters, generation)

        if verbose:
            print(
                f"{generation}\t{parameters.min_fitnesses[generation]}\t{parameters.avg_fitnesses[generation]}\t{parameters.sd_fitnesses[generation]}")

        if min(fitnesses_list) <= parameters.min_assumable_fitness:
            success = True
            break


    # Prints info of the best individual
    best_individual = population[np.argmin(fitnesses_list)]
    print(f"Best individual: {decode_individual(best_individual, parameters)[0]}, fitness value: {best_individual.fitness.values[0]}")

    return parameters, success, best_individual.fitness.values[0], number_of_evaluations
