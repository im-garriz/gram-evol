"""
File where adaptative restrictions constant are updated
"""

def manage_adaptative_restrictions(parameters):

    """
    Function that updates the constant corresponding to the restrictions of the algorithm

    :param parameters: parameters of the algorithm

    :return: recalculate (True if some constant is changed, as individuals must be re-evaluated when this happens,
             False otherwise)
    """

    recalculate = False

    if parameters.best_individual_factible_times_populationLen == parameters.Nf:
        parameters.lambda_genotype_len = max(parameters.min_lambda,
                                             (1/parameters.beta2) * parameters.lambda_genotype_len)
        recalculate = True
    elif parameters.best_individual_factible_times_populationLen == 0:
        parameters.lambda_genotype_len = min(parameters.max_lambda_len,
                                             parameters.beta1 * parameters.lambda_genotype_len)
        recalculate = True

    parameters.best_individual_factible_times_populationLen = 0

    if parameters.best_individual_factible_integrationConst == parameters.Nf:
        parameters.lambda_integration_const = max(parameters.min_lambda,
                                                  (1 / parameters.beta2) * parameters.lambda_integration_const)
        recalculate = True
    elif parameters.best_individual_factible_integrationConst == 0:
        parameters.lambda_integration_const = min(parameters.max_lambda_integration_const,
                                                  parameters.beta1 * parameters.lambda_integration_const)
        recalculate = True

    parameters.best_individual_factible_integrationConst = 0

    return recalculate
