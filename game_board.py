import numpy as np

np.random.seed(111)

NUM_ROWS = NUM_COLS = 9
MIN_VAL = 1
MAX_VAL = 9


# sum automaton class
class candidate:

    def __init__(self, board):
        self.board = board
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
            existing_vals = np.unique(self.board[:, i])  # get unique values in the row
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

    def check_validity(self):

       xxx = 1