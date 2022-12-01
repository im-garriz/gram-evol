from GE import run_GE
import numpy as np
import random
from parameters import Parameters

"""
File that runs the algorithm N_OF_INDEPENDENT_EXECUTIONS independent times and genrated the log
"""

def main():

    LOG_FILE_NAME = "problem_6.csv"
    RANDOM_SEED = 42
    N_OF_INDEPENDENT_EXECUTIONS = 30
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    number_of_successes = 0
    VAMs = []
    n_of_evaluations = []

    # Initializates parameters
    parameters = Parameters()

    # Initialized log lists
    avg_fitness_progress_curve = np.zeros((1, parameters.max_gens+1))
    min_fitness_progress_curve = np.zeros((1, parameters.max_gens+1))
    std_fitness_progress_curve = np.zeros((1, parameters.max_gens+1))

    avg_length_progress_curve = np.zeros((1, parameters.max_gens+1))
    min_length_progress_curve = np.zeros((1, parameters.max_gens+1))
    max_length_progress_curve = np.zeros((1, parameters.max_gens+1))

    avg_wrapping_progress_curve = np.zeros((1, parameters.max_gens+1))
    std_wrapping_progress_curve = np.zeros((1, parameters.max_gens+1))

    for _ in range(N_OF_INDEPENDENT_EXECUTIONS):

        print(f"Execution number {_}")
        parameters, success, best_individuals_fitness_value, number_of_evaluations = run_GE(parameters, verbose=False)

        if success:
            number_of_successes += 1
            n_of_evaluations.append(number_of_evaluations)

        VAMs.append(best_individuals_fitness_value)

        avg_fitness_progress_curve = np.sum((avg_fitness_progress_curve,
                                             np.array(parameters.avg_fitnesses).reshape(1, parameters.max_gens+1)), axis=0)
        min_fitness_progress_curve = np.sum((min_fitness_progress_curve,
                                             np.array(parameters.min_fitnesses).reshape(1, parameters.max_gens+1)), axis=0)
        std_fitness_progress_curve = np.sum((std_fitness_progress_curve,
                                             np.array(parameters.sd_fitnesses).reshape(1, parameters.max_gens+1)), axis=0)

        avg_length_progress_curve = np.sum((avg_length_progress_curve,
                                            np.array(parameters.avg_lens).reshape(1, parameters.max_gens+1)), axis=0)
        min_length_progress_curve = np.sum((min_length_progress_curve,
                                            np.array(parameters.min_lens).reshape(1, parameters.max_gens+1)), axis=0)
        max_length_progress_curve = np.sum((max_length_progress_curve,
                                            np.array(parameters.max_lens).reshape(1, parameters.max_gens+1)), axis=0)

        avg_wrapping_progress_curve = np.sum((avg_wrapping_progress_curve,
                                              np.array(parameters.avg_wrapping).reshape(1, parameters.max_gens+1)), axis=0)
        std_wrapping_progress_curve = np.sum((std_wrapping_progress_curve,
                                              np.array(parameters.sd_wrapping).reshape(1, parameters.max_gens+1)), axis=0)


    TE = 100.0 * number_of_successes / N_OF_INDEPENDENT_EXECUTIONS
    VAMM = np.mean(VAMs)
    if len(n_of_evaluations) > 0:
        PEX = np.mean(n_of_evaluations)
    else:
        PEX = -1

    print(f"Generating log file: {LOG_FILE_NAME}")

    avg_fitness_progress_curve /= N_OF_INDEPENDENT_EXECUTIONS
    min_fitness_progress_curve /= N_OF_INDEPENDENT_EXECUTIONS
    std_fitness_progress_curve /= N_OF_INDEPENDENT_EXECUTIONS
    avg_length_progress_curve /= N_OF_INDEPENDENT_EXECUTIONS
    min_length_progress_curve /= N_OF_INDEPENDENT_EXECUTIONS
    max_length_progress_curve /= N_OF_INDEPENDENT_EXECUTIONS
    avg_wrapping_progress_curve /= N_OF_INDEPENDENT_EXECUTIONS
    std_wrapping_progress_curve /= N_OF_INDEPENDENT_EXECUTIONS

    with open(f"../log/{LOG_FILE_NAME}", 'w') as file:

        file.write("generation,avg_fitness,min_fitness,std_fitness,avg_genotype_len,min_genotype_len,max_genotype_len,avg_wrapping,std_wrapping\n")

        for g in range(avg_fitness_progress_curve.shape[1]):
            file.write(f"{g},{avg_fitness_progress_curve[0, g]},{min_fitness_progress_curve[0, g]},{std_fitness_progress_curve[0, g]},{avg_length_progress_curve[0, g]},{min_length_progress_curve[0, g]},{max_length_progress_curve[0, g]},{avg_wrapping_progress_curve[0, g]},{std_wrapping_progress_curve[0, g]}\n")

        file.write(f"#TE: {TE}\n")
        file.write(f"#VAMM: {VAMM}\n")
        file.write(f"#PEX: {PEX}\n")

    print(f"{LOG_FILE_NAME} succesfully generated")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit()