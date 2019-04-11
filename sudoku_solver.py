
import numpy as np
import operator
import game_board as gb


np.random.seed(111)

NUM_SOL = 100
NUM_REPLICATIONS = 0.25 * NUM_SOL  # number of replications
MUTATION_PERC = 0.05  # % of mutations
NUM_ELITISM = 0.05 * NUM_SOL  # number of best solutions to take

# generate initial solutions from user input
def generate_initial_solutions(input_board):

    solutions = []
    for i in range(NUM_SOL):
        board = gb.candidate(input_board)
        board.draw_random_solution()
        solutions.append(board)
    return solutions


# get user input
def get_user_input():

    valid_input = False

    # get the input is not in a valid form
    while not valid_input:

        # get initial board
        for i in range(gb.NUM_ROWS):
            row = input()
            try:
                numeric_row = np.asarray([int(s) for s in row.split(' ')])
                # if the input length is different than 9 or there is a number not in the defined range start again
                if len(numeric_row) != gb.NUM_ROWS or any(numeric_row < 0) or any(numeric_row > gb.MAX_VAL):
                    print("You entered an invalid value! Please try again")
                    valid_input = False
                    break

                valid_input = True

            # in case there was a problem with casting to int
            except ValueError:
                print("You entered an invalid value! Please try again")
                valid_input = False
                break

            if valid_input:
                board = numeric_row.reshape(1, -1) if i == 0 else np.concatenate((board, numeric_row.reshape(1, -1)))

    return board


# set the fitness of each solution
def set_fitness(solutions):
    all_fitness = np.asarray([])
    # calc per each solution the fitness and save it
    for sol in solutions:
        sol.set_fitness()
        all_fitness = np.append(all_fitness, np.asarray(sol.get_fitness()))
    f_mean = np.mean(all_fitness)  # mean fitness
    f_max = np.max(all_fitness)  # max fitness
    f_min = np.min(all_fitness)  # min fitness
    f_sum = np.sum(all_fitness)
    return f_sum, f_mean, f_max, f_min


# get probabilities based on the fitnesses
def from_fitness_to_probs(solutions, f_sum):
    for sol in solutions:
        sol.set_prob(f_sum)


# select solutions based on biased selection
def select_solutions_for_replications(solutions):

    sol_probs = []  # save probability of each solution
    # get probabilities while relying on the fact that they are sorted
    for sol in solutions:
        sol_probs.append(sol.get_prob())

    sol_indices = np.arange(NUM_SOL)  # indices of the solutions

    # sample solutions according to their probabilities without replacement
    sampled_indices = np.random.choice(sol_indices, size=int(NUM_REPLICATIONS), replace=False, p=sol_probs)
    return sorted(sampled_indices.tolist())


# game course
def play_sudoku():
    input_board = get_user_input()  # get initial board
    solutions = generate_initial_solutions(input_board)  # generate 100 solutions
    f_sum, f_mean, f_max, f_min = set_fitness(solutions)  # calc the fitness of each solution
    best_fitness = gb.NUM_ROWS * 3  # the best score is 27
    from_fitness_to_probs(solutions, f_sum)  # set the probability of each solution
    solutions.sort(key=operator.attrgetter('prob'), reverse=True)  # sort instances by the probability

    next_solutions = []
    # as long as there isn't a valid solution
    while f_max < best_fitness:
        rep_sols = select_solutions_for_replications(solutions)  # get solutions from replications
        elit_sols = list(range(int(NUM_ELITISM)))  # get elite solutions
        sols_indx_proceeding_next_gen = set(rep_sols + elit_sols)  # merge replication and elitism solutions
        sols_proceeding_next_gen = [solutions[i] for i in sols_indx_proceeding_next_gen]  # get the solutions that stay as is for next generation
        



if __name__ == '__main__':

    play_sudoku()
