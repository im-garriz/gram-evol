import numpy as np
from decode_backus_naur import decode_individual
import math

"""
File that contain all corresponding to the fitness value calculation
"""

def evaluate_individual(individual, parameters):

    """
    Evaluates an individual

    :param individual: individual to evaluate
    :param parameters: paraemters of the algorithm

    :return: (fitness_value, True if satisfies the integration constant condition; False otherwise)
    """

    deltaX = (parameters.interval[1] - parameters.interval[0]) / parameters.N

    sum = 0

    # Decodes the mathematical expression corresponding to the individual
    decoded_function, correctly_decoded = decode_individual(individual, parameters)

    if not correctly_decoded:
        return parameters.fitness_for_invalid_individuals, False

    for i in range(parameters.N+1):

        x = parameters.interval[0] + i * deltaX
        fx = get_value_function_to_integrate(x, parameters)
        Fhat_derived_x, division_by_zero = get_value_decoded_function(x, decoded_function, parameters)

        if division_by_zero:
            return parameters.fitness_for_invalid_individuals, False

        omegai = parameters.K1
        absolute_difference = abs(Fhat_derived_x - fx)
        if absolute_difference <= parameters.U:
            omegai = parameters.K0

        sum += omegai * absolute_difference

    if sum > parameters.fitness_for_invalid_individuals:
        return parameters.fitness_for_invalid_individuals, False

    fitness_val = (sum / (parameters.N + 1))

    genotype_len_penalty = get_genotype_len_penalty(individual, parameters)
    integration_constant_penalty = get_integration_constant_penalty(decoded_function, parameters)

    return (fitness_val + genotype_len_penalty + integration_constant_penalty,
            integration_constant_penalty == 0)


def get_value_function_to_integrate(x, parameters):

    """
    Function that computes the value of the function to integrate in the point x

    :param x: point
    :param parameters: parameters of the algorithm

    :return: f(x)
    """

    if parameters.problem == 1:
        return 6 * x**2
    elif parameters.problem == 2:
        return 2 / ((x + 1) ** 2)
    elif parameters.problem == 3:
        return (3 * x**2 - 2*x + 1)/4
    elif parameters.problem == 4:
        return (np.exp(2 * x) - np.exp(-6 * x))/3
    elif parameters.problem == 5:
        return np.log(1 + x) + (x / (1 + x))
    else:
        return np.exp(x) * (np.sin(x) + np.cos(x))


def get_value_decoded_function(x, decoded_function, parameters):

    """
    Evaluates the integral of the decoded expression (from the codon chain) in the point x, thus,
    returns (f^(x+h) - f^(x)) / h (where f^ is the decoded expression)

    :param x: point
    :param decoded_function: decoded expression
    :param parameters: parameters of the algorithm

    :return: (f^(x+h) - f^(x)) / h
    """

    try:
        fx = eval(f"{decoded_function}")
        x += parameters.h
        fx_plus_h = eval(f"{decoded_function}")

    except ZeroDivisionError:
        return 0, True

    except ValueError:
        return 1, True

    except OverflowError:
        return 2, True
    except:
        return 3, True

    return (fx_plus_h - fx) / parameters.h, False


def get_genotype_len_penalty(individual, parameters):

    """
    Function that returns the genotype length penalty component of the fitness value

    :param individual: individual to evaluate
    :param parameters: parameters of the algorithm

    :return: genotype length penalty component
    """

    gx = max(0, len(individual) - parameters.max_genotype_len)
    return min(parameters.lambda_genotype_len * gx,
               parameters.fitness_for_invalid_individuals)


def get_integration_constant_penalty(decoded_function, parameters):

    """
        Function that returns the integration constant penalty component of the fitness value

        :param individual: individual to evaluate
        :param parameters: parameters of the algorithm

        :return: integration constant penalty component
        """

    x = 0
    hx = abs(eval(decoded_function) - parameters.F_0)

    return min(parameters.lambda_integration_const * max(0, hx - parameters.epsilon_integration_constant_tol),
               parameters.fitness_for_invalid_individuals)

