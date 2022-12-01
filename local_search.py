from decode_backus_naur import get_non_terminal_indexes
import numpy as np

"""
File where local search operator is implemented
"""


def local_search(individual, parameters, toolbox):

    """
    Local search operator. Performs a local search on the provided individual

    :param individual: individual
    :param parameters: parameters of the algorithm
    :param toolbox: toolbox of the algorith (deap object)

    :return: new individual
    """

    n_of_evaluations = 0
    if np.random.random() <= parameters.local_search_prob:

        terminal_idxs, correctly_decoded = get_non_terminal_indexes(individual, parameters)

        if len(terminal_idxs) > 0:
            if correctly_decoded:
                n_codons_to_modify = min(len(terminal_idxs), parameters.n_codons_to_modify)

                idxs_to_vary = terminal_idxs[-n_codons_to_modify:]

                neighborhood = []
                for idx in idxs_to_vary:
                    for i in range(1, 4):
                        neighbor = toolbox.clone(individual)
                        neighbor[idx] += i
                        neighborhood.append(neighbor)

                neighborhood_fitnesses = [toolbox.evaluate(individual)[0] for individual in neighborhood]
                n_of_evaluations += len(neighborhood)

                if not(individual.fitness.values[0] <= min(neighborhood_fitnesses)):
                    individual_to_return = neighborhood[np.argmin(neighborhood_fitnesses)]
                    individual_to_return.fitness.values = min(neighborhood_fitnesses),
                    return individual_to_return, n_of_evaluations

    return toolbox.clone(individual), n_of_evaluations
