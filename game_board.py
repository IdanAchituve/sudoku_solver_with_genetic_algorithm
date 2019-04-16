import numpy as np

np.random.seed(222)

NUM_ROWS = NUM_COLS = 9
MIN_VAL = 1
MAX_VAL = 9


# sum automaton class
class Candidate:

    def __init__(self, board):
        self.board = board.copy()
        self.fixed_board = board.copy()  # save the fixed cells of the board
        self.fitness = 0
        self.prob = 0

    # draw random numbers in the board gaps such that each row will not have duplicates
    def draw_random_solution(self):
        valid_rows_vals = np.asarray(list(range(MIN_VAL, MAX_VAL + 1)))  # get valid values for each rows
        for i in range(NUM_ROWS):
            existing_vals = np.unique(self.board[i])  # get unique values in the row
            vals_to_fill = np.setdiff1d(np.union1d(valid_rows_vals, np.asarray([0])), existing_vals)  # get values to fill beside 0
            np.random.shuffle(vals_to_fill)
            np.place(self.board[i], self.board[i] == 0, vals_to_fill)  # fill the missing values

    # fitness = #rows with all values in range + #cols with all values in range + #squares with all values in range
    def set_fitness(self):
        valid_rows_vals = np.asarray(list(range(MIN_VAL, MAX_VAL + 1)))  # get valid values for each rows
        fitness = 0
        # check rows
        for i in range(NUM_ROWS):
            existing_vals = np.unique(self.board[i])  # get unique values in the row
            if len(np.setdiff1d(valid_rows_vals, existing_vals)) == 0:
                fitness += 1

        # check cols
        for i in range(NUM_COLS):
            existing_vals = np.unique(self.board[:, i])  # get unique values in the column
            if len(np.setdiff1d(valid_rows_vals, existing_vals)) == 0:
                fitness += 1

        # check squares
        step_size = int(np.sqrt(NUM_ROWS))
        for i in range(0, NUM_ROWS, step_size):
            for j in range(0, NUM_COLS, step_size):
                existing_vals = np.unique(self.board[i:i+step_size, j:j+step_size]).flatten()  # get unique values in the row
                if len(np.setdiff1d(valid_rows_vals, existing_vals)) == 0:
                    fitness += 1

        self.fitness = fitness

    def get_fitness(self):
        return self.fitness

    def set_prob(self, f_sum):
        self.prob = self.fitness/f_sum

    def get_prob(self):
        return self.prob

    def get_board(self):
        return self.board

    def set_board(self, board):
        self.board = board

    def mutate(self, mutation_rate):

        # per each row generate random number and do mutation
        rand_num = np.random.random()
        mutated = False

        # do mutation upon success and if it is possible to improve
        self.set_fitness()
        if rand_num < mutation_rate and self.fitness < MAX_VAL * 3:

            while not mutated:
    
                # pick a row and then pick to columns. By doing so I maintain that each row will not have duplicates
                row = np.random.randint(0, NUM_ROWS)
                cols = np.random.randint(0, NUM_COLS, 2)
                while self.fixed_board[row, cols[0]] != 0 or self.fixed_board[row, cols[1]] != 0 or cols[0] == cols[1]:
                    row = np.random.randint(0, NUM_ROWS)
                    cols = np.random.randint(0, NUM_COLS, 2)
    
                val1 = self.board[row, cols[0]]  # 1st value to switch
                val2 = self.board[row, cols[1]]  # 2nd value to switch
    
                # switch values
                self.board[row, cols[0]] = val2
                self.board[row, cols[1]] = val1
                mutated = True


    def optimization(self):
        xxx = 1