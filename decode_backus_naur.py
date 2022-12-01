"""
File that contain all the operators corresponding to individual decodification
"""

def decode_individual(individual, parameters):

    """
    Function that decodes an individutal into a mathematical expression

    :param individual: individual to decode
    :param parameters: parameters of the algorithm

    :return: decoded_expression (str), successful_decoding (boolean)
    """

    # Initial expression
    S = ["<expr>"]

    decoded_expression = S[0]

    current_codon_idx = 0
    n_wrapping = 0
    continue_decoding = True

    while continue_decoding:

        # Finds the first non terminal in the current expression
        first_non_terminal = find_first_non_terminal(decoded_expression)

        if first_non_terminal == "":
            parameters.wrappings_by_individual.append(n_wrapping)
            return decoded_expression, True

        # Gets the producion rules corresponding to the first non-terminal expression and replaces it with
        # a new expression
        production_rules = get_prodution_rules_for_non_terminal(first_non_terminal)
        n_of_rules = len(production_rules)
        current_codon = individual[current_codon_idx]
        rule_idx = current_codon % n_of_rules
        rule = production_rules[rule_idx]
        decoded_expression = decoded_expression.replace(first_non_terminal, rule, 1)

        current_codon_idx += 1

        if current_codon_idx >= len(individual):
            n_wrapping += 1
            current_codon_idx = 0

        if n_wrapping > parameters.max_wrapping:
            parameters.wrappings_by_individual.append(n_wrapping)
            continue_decoding = False

    return "", False


def find_first_non_terminal(expression):

    """
    Returns the first non-terminal in the expression

    :param expression: expression where non-terminals are being search

    :return: first non-terminal ("", if the is not any)
    """

    for i, char in enumerate(expression):

        if char == "<":
            first_idx = i
            last_idx = i

            while expression[last_idx] != ">":
                last_idx += 1

            return expression[first_idx:last_idx+1]

    return ""


def get_prodution_rules_for_non_terminal(non_terminal):

    """
    Returs the list of production rules of a certain non-terminal

    :param non_terminal: non-terminal

    :return: production rules
    """

    expr = ["<expr><op><expr>", "(<expr><op><expr>)", "<pre_op>(<expr>)", "<var>"]
    op = ["+", "-", "*", "/"]
    pre_op = ["math.sin", "math.cos", "math.exp", "math.log"]
    var = ["x", "<integer>"]
    integer = ["1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0"]

    if non_terminal == "<expr>":
        return expr
    elif non_terminal == "<op>":
        return op
    elif non_terminal == "<pre_op>":
        return pre_op
    elif non_terminal == "<var>":
        return var
    elif non_terminal == "<integer>":
        return integer
    else:
        raise Exception('Trying to decode an invalid non terminal expression: {}'.format(non_terminal))


def get_non_terminal_indexes(individual, parameters):

    """
    Function that returns a list with the indexes of the terminal codons in the individual

    :param individual: individual where to find
    :param parameters: parameters of the algorithm

    :return: list of terminal codons indexes
    """

    S = ["<expr>"]

    decoded_expression = S[0]

    current_codon_idx = 0
    n_wrapping = 0
    continue_decoding = True

    terminal_idx_list = []

    while continue_decoding:

        first_non_terminal = find_first_non_terminal(decoded_expression)

        if first_non_terminal == "":
            parameters.wrappings_by_individual.append(n_wrapping)
            return list(set(terminal_idx_list)), True

        if first_non_terminal != "<expr>" and first_non_terminal != "<var>":
            terminal_idx_list.append(current_codon_idx)

        production_rules = get_prodution_rules_for_non_terminal(first_non_terminal)
        n_of_rules = len(production_rules)

        current_codon = individual[current_codon_idx]

        rule_idx = current_codon % n_of_rules
        rule = production_rules[rule_idx]

        decoded_expression = decoded_expression.replace(first_non_terminal, rule, 1)

        current_codon_idx += 1

        if current_codon_idx >= len(individual):
            n_wrapping += 1
            current_codon_idx = 0

            if n_wrapping > parameters.max_wrapping:
                parameters.wrappings_by_individual.append(n_wrapping)
                continue_decoding = False

    return list(set(terminal_idx_list)), False
