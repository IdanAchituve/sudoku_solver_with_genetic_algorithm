
import numpy as np
import operator
import copy
import datetime
import game_board as gb


np.random.seed(222)

NUM_SOL = 100
NUM_REPLICATIONS = 0.1 * NUM_SOL  # number of replications
NUM_ELITISM = 0.01 * NUM_SOL  # number of best solutions to take
MUTATION_RATE = 0.2
MAX_STAGNATION = 1000  # max number of generations with the same max solution
FITTEST_VS_WEAKEST = 0.85


# generate initial solutions from user input
def generate_initial_solutions(input_board):

    solutions = []
    for i in range(NUM_SOL):
        board = gb.Candidate(input_board.copy())
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
def bias_selection(solutions, sample_size=int(NUM_REPLICATIONS)):

    sol_probs = []  # save probability of each solution
    # get probabilities while relying on the fact that they are sorted
    for sol in solutions:
        sol_probs.append(sol.get_prob())

    sol_indices = np.arange(len(solutions))  # indices of the solutions

    # sample solutions according to their probabilities without replacement
    sampled_indices = np.random.choice(sol_indices, size=sample_size, replace=False, p=sol_probs)
    return sorted(sampled_indices.tolist())


# mix 2 parent rows according to: Learning Bayesian network structures by searching for the best ordering with genetic algorithms. IEEE Transactions on System, Man and Cybernetics, 26(4):487-493
def cyclic_crossover(prow1, prow2):

    # new row values
    crow1 = np.zeros(gb.MAX_VAL)
    crow2 = np.zeros(gb.MAX_VAL)

    left = list(range(gb.MAX_VAL))
    cycle = 0

    for i in range(gb.MAX_VAL):

        # already assigned this values
        if i not in left:
            continue

        if cycle % 2 == 0:
            start_val = prow1[i]  # value of the 1st parent at the ith position
            p2val = prow2[i]  # value of the 2nd parent at the ith position
            crow1[i] = start_val  # 1st child with 1st parent values at the ith position
            crow2[i] = p2val  # 2nd child with the 2nd parent value at the uth position
            left.remove(i)

            while start_val != p2val:
                p1index = np.where(prow1 == p2val)[0].item()
                crow1[p1index] = prow1[p1index]  # get value in the 1st parent corresponding to the index equal to the value of the 2nd parent
                p2val = prow2[p1index]
                crow2[p1index] = p2val
                left.remove(p1index)

            cycle += 1

        else:
            start_val = prow1[i]  # value of the 1st parent at the ith position
            p2val = prow2[i]  # value of the 2nd parent at the ith position
            crow1[i] = p2val  # 1st child with 2nd parent values at the ith position
            crow2[i] = start_val  # 2nd child with the 1st parent value at the uth position
            left.remove(i)

            while start_val != p2val:
                p1index = np.where(prow1 == p2val)[0].item()
                p2val = prow2[p1index]
                crow1[p1index] = p2val  # get value in the 1st parent corresponding to the index equal to the value of the 2nd parent
                crow2[p1index] = prow1[p1index]
                left.remove(p1index)

            cycle += 1

    return crow1, crow2


# perform cycle crossover on rows
def cross_over(solutions, num_cross_over_sols):

    # get 2 solutions to compete against each other
    def tournament(p1_index, p2_index):
        p1_fitness = solutions[p1_index].get_fitness()
        p2_fitness = solutions[p2_index].get_fitness()
        p1_higher = p1_fitness >= p2_fitness
        r = np.random.rand()
        if (r < FITTEST_VS_WEAKEST and p1_higher) or (r >= FITTEST_VS_WEAKEST and not p1_higher):
            return p1_index
        return p2_index


    cross_over_solutions = []
    for i in range(int(num_cross_over_sols)):

        #  choose 2 parents based on tournament
        parent1 = tournament(np.random.randint(len(solutions)), np.random.randint(len(solutions)))
        parent2 = tournament(np.random.randint(len(solutions)), np.random.randint(len(solutions)))

        child1 = copy.deepcopy(solutions[parent1])  # get the 1st parent
        child2 = copy.deepcopy(solutions[parent2])  # get the 2nd parent
        child1_board = child1.get_board().copy()
        child2_board = child2.get_board().copy()

        # select randomly a range of rows to perform crossover on
        start_row = end_row = 0
        while start_row >= end_row:
            start_row = np.random.randint(0, gb.NUM_ROWS)
            end_row = np.random.randint(0, gb.NUM_ROWS)

        # mix row1 and row2 values so that each row will be a valid row (i.e., all values between 1-9)
        for row in range(start_row, end_row):
            child1_board[row, :], child2_board[row, :] = cyclic_crossover(child1_board[row, :], child2_board[row, :])

        # update board for next generation
        child1.set_board(child1_board)
        child2.set_board(child2_board)
        cross_over_solutions.append(child1)
        cross_over_solutions.append(child2)

    return cross_over_solutions


# game course
def play_sudoku():
    input_board = get_user_input()  # get initial board
    solutions = generate_initial_solutions(input_board)  # generate 100 solutions
    f_sum, f_mean, f_max, f_min = set_fitness(solutions)  # calc the fitness of each solution
    best_fitness = gb.NUM_ROWS * 3  # the best score is 27
    from_fitness_to_probs(solutions, f_sum)  # set the probability of each solution
    solutions.sort(key=operator.attrgetter('prob'), reverse=True)  # sort instances by the probability

    # as long as there isn't a valid solution or didn't reach the number of generations limit
    generation = fitness_calls = min_max = seq_max = switches = 0
    best_of_breed = []
    while f_max < best_fitness:

        # create new solutions for next generation
        rep_sols = bias_selection(solutions) if NUM_REPLICATIONS > 0 else []  # get solutions from replications
        elit_sols = list(range(int(NUM_ELITISM)))  # get elite solutions
        sols_indx_proceeding_next_gen = list(set(rep_sols + elit_sols))  # get indices of replication and elitism solutions
        # if the number of solutions achieved by replication and elitism is not even remove a replication solution
        if len(sols_indx_proceeding_next_gen) % 2 != 0:
            sols_indx_proceeding_next_gen = sols_indx_proceeding_next_gen[:-1]
        replication_sol = [copy.deepcopy(solutions[i]) for i in sols_indx_proceeding_next_gen if i >= NUM_ELITISM]  # replication
        elitism_sol = [copy.deepcopy(solutions[i]) for i in sols_indx_proceeding_next_gen if i < NUM_ELITISM]  # elitism
        cross_over_sol = cross_over(solutions, int((NUM_SOL - len(sols_indx_proceeding_next_gen))/2))  # crossover

        #  mutation
        for sol in replication_sol + cross_over_sol:
            sol.mutate(MUTATION_RATE)

        # update to the new solutions
        solutions = replication_sol + elitism_sol + cross_over_sol  # create new solution list

        prev_f_max = f_max
        f_sum, f_mean, f_max, f_min = set_fitness(solutions)  # calc the fitness of each solution
        from_fitness_to_probs(solutions, f_sum)  # set the probability of each solution
        solutions.sort(key=operator.attrgetter('prob'), reverse=True)  # sort instances by the probability

        best_sol = copy.deepcopy(solutions[0]) if generation == 0 else best_sol if solutions[0].get_fitness() < best_sol.get_fitness() else copy.deepcopy(solutions[0])

        # On convergence seed new solutions
        min_max = min_max + 1 if f_max == f_min else 0
        seq_max = seq_max + 1 if f_max == prev_f_max else 0
        if seq_max >= MAX_STAGNATION or min_max >= MAX_STAGNATION:

            # if even the best of breed solutions didn't help - quit
            if switches == 11:
                break

            # save the top 10% solutions from each restart
            for sol_idx, sol in enumerate(solutions):
                if sol_idx < len(solutions)/10:
                    best_of_breed.append(copy.deepcopy(sol))

            solutions = generate_initial_solutions(input_board) if switches < 10 else best_of_breed
            f_sum, f_mean, f_max, f_min = set_fitness(solutions)  # calc the fitness of each solution
            from_fitness_to_probs(solutions, f_sum)  # set the probability of each solution
            solutions.sort(key=operator.attrgetter('prob'), reverse=True)  # sort instances by the probability

            min_max = seq_max = 0
            switches += 1

        # print progress
        if generation % 100 == 0:
            datetime_string = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")
            print(datetime_string + "\tbreed: " + str(switches) + "\tgeneration: " + str(generation) + "\t\tmean fitness: " + str(f_mean) +
                  "\tmin fitness: " + str(f_min) + "\tmax fitness: " + str(f_max))

        # update variables
        generation += 1
        fitness_calls = NUM_SOL * generation

    # print best solution
    board = best_sol.get_board()
    print(board)
    print("Number of calls: " + str(fitness_calls))


if __name__ == '__main__':

    play_sudoku()