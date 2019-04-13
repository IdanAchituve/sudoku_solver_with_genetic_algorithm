import numpy as np

np.random.seed(222)

NUM_ROWS = NUM_COLS = 9
MIN_VAL = 1
MAX_VAL = 9
MAX_MUTATUION_TRIES = 3000

# sum automaton class
class candidate:

    def __init__(self, board):
        self.board = board.copy()
        self.fixed_board = board.copy()  # save the fixed cells of the board
        self.fitness = 0
        self.prob = 0

    # draw random numbers in the board gaps such that each row will not have duplicates
    def draw_random_solution(self):
        fill_type = np.random.randint(0, 3)
        if fill_type == 0:
            valid_rows_vals = np.asarray(list(range(MIN_VAL, MAX_VAL + 1)))  # get valid values for each rows
            for i in range(NUM_ROWS):
                existing_vals = np.unique(self.board[i])  # get unique values in the row
                vals_to_fill = np.setdiff1d(np.union1d(valid_rows_vals, np.asarray([0])), existing_vals)  # get values to fill beside 0
                np.random.shuffle(vals_to_fill)
                np.place(self.board[i], self.board[i] == 0, vals_to_fill)  # fill the missing values
        elif fill_type == 1:
            valid_rows_vals = np.asarray(list(range(MIN_VAL, MAX_VAL + 1)))  # get valid values for each rows
            for i in range(NUM_COLS):
                existing_vals = np.unique(self.board[:, i])  # get unique values in the row
                vals_to_fill = np.setdiff1d(np.union1d(valid_rows_vals, np.asarray([0])), existing_vals)  # get values to fill beside 0
                np.random.shuffle(vals_to_fill)
                np.place(self.board[:, i], self.board[:, i] == 0, vals_to_fill)  # fill the missing values
        else:
            valid_rows_vals = np.asarray(list(range(MIN_VAL, MAX_VAL + 1)))  # get valid values for each rows
            step_size = int(np.sqrt(NUM_ROWS))
            for i in range(0, NUM_ROWS, step_size):
                for j in range(0, NUM_COLS, step_size):
                    existing_vals = np.unique(self.board[i:i + step_size, j:j + step_size]).flatten()  # get unique values in the row
                    vals_to_fill = np.setdiff1d(np.union1d(valid_rows_vals, np.asarray([0])), existing_vals)  # get values to fill beside 0
                    np.random.shuffle(vals_to_fill)
                    np.place(self.board[i:i + step_size, j:j + step_size], self.board[i:i + step_size, j:j + step_size] == 0, vals_to_fill)  # fill the missing values

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

    def get_fixed_board(self):
        return self.fixed_board

    def mutate(self, mutation_rate, mutation_type_rate):

        # do mutation upon success and if it is possible to improve
        rand_num = np.random.random()
        mutated = False

        if rand_num < mutation_rate and self.fitness < MAX_VAL * 3:
            mutation_type = np.random.random()

            if mutation_type < mutation_type_rate:
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
            else:
                """
                rand_row = np.random.randint(0, NUM_ROWS)
                valid_rows_vals = np.asarray(list(range(MIN_VAL, MAX_VAL + 1)))  # get valid values for each rows
                existing_vals = np.unique(self.fixed_board[rand_row])  # get unique values in the row
                vals_to_fill = np.setdiff1d(np.union1d(valid_rows_vals, np.asarray([0])), existing_vals)  # get values to fill beside 0
                np.random.shuffle(vals_to_fill)
                idx = 0
                for col in range(NUM_COLS):
                    if self.fixed_board[rand_row, col] == 0:
                        self.board[rand_row, col] = vals_to_fill[idx]
                        idx += 1

                """
                rand_num = np.random.random()
                valid_rows_vals = np.asarray(list(range(MIN_VAL, MAX_VAL + 1)))  # get valid values
                if rand_num < 0.5:
                    rand_col = np.random.randint(0, NUM_COLS)
                    unique, counts = np.unique(self.board[:, rand_col], return_counts=True)
                    num_counts = dict(zip(unique, counts))

                    if len(num_counts.keys()) < MAX_VAL:
                        missings_vals = np.setdiff1d(valid_rows_vals, np.asarray(list(num_counts.keys())))
                        val_to_fill = np.random.choice(missings_vals)
                        rand_row = np.random.randint(0, NUM_COLS)
                        while self.fixed_board[rand_row, rand_col] != 0:
                            rand_row = np.random.randint(0, NUM_COLS)

                        self.board[rand_row, rand_col] = val_to_fill
                else:
                    rand_box = np.random.randint(0, int(NUM_ROWS/3), 2)
                    box_row = 3 * rand_box[0]
                    box_col = 3 * rand_box[1]
                    unique, counts = np.unique(self.board[box_row:box_row+3, box_col:box_col+3], return_counts=True)
                    num_counts = dict(zip(unique, counts))

                    if len(num_counts.keys()) < MAX_VAL:
                        missings_vals = np.setdiff1d(valid_rows_vals, np.asarray(list(num_counts.keys())))
                        val_to_fill = np.random.choice(missings_vals)
                        rand_row = np.random.randint(box_row, box_row + 3)
                        rand_col = np.random.randint(box_col, box_col + 3)
                        while self.fixed_board[rand_row, rand_col] != 0:
                            rand_row = np.random.randint(box_row, box_row + 3)
                            rand_col = np.random.randint(box_col, box_col + 3)

                        self.board[rand_row, rand_col] = val_to_fill
