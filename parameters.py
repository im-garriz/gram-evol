import numpy as np


class Parameters:

    """
    Class that contains all configuration parameters
    """

    def __init__(self):

        # General parametes
        self.max_genotype_len = 40
        self.population_length = 200
        if self.population_length % 2 != 0:
            self.population_length += 1
        self.max_gens = 1500

        # Initialization parameters
        self.min_initial_len = 10
        self.max_initial_len = 20

        # Parent selection parameters
        self.tournament_size = 5
        self.initial_tournament_size = 5
        self.final_tournament_size = 15

        # Mating parameters
        self.mating_probability = 0.9

        # Mutation parameters
        self.mutation_prob = 0.25
        self.pm_inf = 0.15
        self.pm_0 = 0.75

        # Duplication parameters
        self.min_duplication_len = 0
        self.max_duplication_len = 5
        self.duplication_probability = 0.00

        # Survival selection parameters
        self.survival_selection_method = "steady_state_model" # mu_plus_lambda, generational, steady_state_model
        self.elitism = True

        # Parameters for fitness value calculation
        self.min_assumable_fitness = 0.1
        self.max_wrapping = 5
        self.problem = 1
        self.compute_problem_parameters()

        self.N = -1
        self.N_deltaX_1 = 10
        self.h = 0.00001

        self.U = 0.1
        self.K0 = 1
        self.K1 = 10

        self.fitness_for_invalid_individuals = 500000

        # Parameters of restrictions
        self.Nf = 5
        self.beta1 = 4
        self.beta2 = 2.8

        self.min_lambda = 0.1
        self.max_lambda_len = 10
        self.max_lambda_integration_const = 1

        self.lambda_integration_const = 1
        self.lambda_genotype_len = 1
        self.epsilon_integration_constant_tol = 0.1

        self.best_individual_factible_times_populationLen = 0
        self.best_individual_factible_integrationConst = 0

        # Parameters for local search
        self.local_search_prob = 0.9
        self.n_codons_to_modify = 5

        # Lists to save logs
        self.avg_fitnesses = [0 for _ in range(self.max_gens + 1)]
        self.min_fitnesses = [0 for _ in range(self.max_gens + 1)]
        self.sd_fitnesses = [0 for _ in range(self.max_gens + 1)]

        self.min_lens = [0 for _ in range(self.max_gens + 1)]
        self.max_lens = [0 for _ in range(self.max_gens + 1)]
        self.avg_lens = [0 for _ in range(self.max_gens + 1)]

        self.avg_wrapping = [0 for _ in range(self.max_gens + 1)]
        self.sd_wrapping = [0 for _ in range(self.max_gens + 1)]

        self.wrappings_by_individual = []

        self.compute_N()

    def clear_logs(self):

        """
        Function that clear logs for each independent executions

        :return:
        """
        self.avg_fitnesses = [0 for _ in range(self.max_gens+1)]
        self.min_fitnesses = [0 for _ in range(self.max_gens+1)]
        self.sd_fitnesses = [0 for _ in range(self.max_gens+1)]

        self.min_lens = [0 for _ in range(self.max_gens+1)]
        self.max_lens = [0 for _ in range(self.max_gens+1)]
        self.avg_lens = [0 for _ in range(self.max_gens+1)]

        self.avg_wrapping = [0 for _ in range(self.max_gens+1)]
        self.sd_wrapping = [0 for _ in range(self.max_gens+1)]

        self.TE = -1
        self.VAMM = -1
        self.PEX = -1

    def compute_N(self):

        """
        Computes N based on the interval length

        :return:
        """

        units_in_interval = self.interval[1] - self.interval[0]
        self.N = units_in_interval * self.N_deltaX_1

    def compute_problem_parameters(self):
        """
        Reads the configuration of the selected problem

        :return:
        """
        if self.problem == 1:
            self.interval = (0, 5)
            self.F_0 = 5
        elif self.problem == 2:
            self.interval = (0, 5)
            self.F_0 = -1
        elif self.problem == 3:
            self.interval = (-2, 2)
            self.F_0 = -0.25
        elif self.problem == 4:
            self.interval = (0, 2)
            self.F_0 = 0.3333333
        elif self.problem == 5:
            self.interval = (0, 5)
            self.F_0 = 0
        elif self.problem == 6:
            self.interval = (-2, 2)
            self.F_0 = 0

    def compute_mutation_prob(self, generation):

        """
        Updated mutation probability

        :param generation: current generation
        :return:
        """

        self.mutation_prob = self.pm_inf + (self.pm_0 - self.pm_inf )/(2**(0.2*generation))

    def compute_tournament_size(self, generation):

        """
        Updates tournanemt size

        :param generation: current generation
        :return:
        """

        self.tournament_size = int(self.final_tournament_size + (self.initial_tournament_size - self.final_tournament_size)\
                               /(2 ** (0.5 * generation)))
