import numpy as np

use_composed_rules = True

def rotational_left_shift(num, shift, width):
    """
    Perform a rotational left bitwise shift on a number.

    Args:
        num (int): The number to shift.
        shift (int): The number of positions to shift.
        width (int): The bit-width of the number.

    Returns:
        int: The result of the rotational left shift.
    """
    # Ensure the number fits within the specified width
    num &= (1 << width) - 1
    if shift < 0:
        shift = -shift  # Convert to positive shift for right rotation
        return ((num >> shift) | (num << (width - shift))) & ((1 << width) - 1)
    else:
        return ((num << shift) | (num >> (width - shift))) & ((1 << width) - 1)

def rule_and(*args, **kwargs):
    if len(args) == 1 and isinstance(args[0], list):
        arr = args[0]
    else:
        arr = args
    def f(data):
        return all(f(data) for f in arr)
    return f

def rule_or(*args, **kwargs):
    if len(args) == 1 and isinstance(args[0], list):
        arr = args[0]
    else:
        arr = args
    def f(data):
        return any(f(data) for f in arr)
    return f

def rule_matrix(constants, **kwargs):
    def f(data):
        matrix_3d = np.array(data)
        num_rows = matrix_3d.shape[0]
        return all(np.sum(constants * matrix_3d[row]) == 0 for row in range(num_rows))
    return f

def rule_set_of_values(values, **kwargs):
    if isinstance(values, list):
        values = set(values)
    def f(data):
        if isinstance(data, list):
            data = set(data)
        return data == values
    return f

def constant(i, **kwargs):
    def f(data):
        if use_composed_rules:
            data = np.array(data)
            # add a constant 1 to the end of data in its 3rd dimension
            data = np.concatenate([data, np.ones((data.shape[0], data.shape[1], 1))], axis=2)
            # make sure this attribute is constant, so get a constant matrix of all zeros except for the i-th column
            c1 = np.zeros((3, data.shape[2]))
            c2 = c1.copy()
            c1[0, i] = 1
            c1[1, i] = -1
            c2[1, i] = 1
            c2[2, i] = -1
            matrix_version = rule_and(list(rule_matrix(c) for c in [c1, c2]))
            return matrix_version(data)
        if len(np.array(data).shape) == 3:
            grid = data
            # we have a 2d matrix of embeddings
            num_rows = np.array(grid).shape[0]
            return all(all(grid[row_i][col_i][i] == grid[row_i][0][i] for col_i in range(3)) for row_i in range(num_rows))
        row = data
        return row[0][i] == row[1][i] == row[2][i]
    return f
def progression(i, is_bitwise=False, **kwargs):
    def f(data):
        if use_composed_rules:
            if is_bitwise:
                # Either is a shift left or a shift right
                num_rows = len(data)
                return any(all(rotational_left_shift(data[row_i][0][i], shift, width) == data[row_i][1][i] and rotational_left_shift(data[row_i][1][i], shift, width) == data[row_i][2][i] for row_i in range(num_rows)) for width in [4, 9] for shift in [1, -1, 2, -2])
            data = np.array(data)
            # add a constant 1 to the end of data in its 3rd dimension
            data = np.concatenate([data, np.ones((data.shape[0], data.shape[1], 1))], axis=2)
            # make sure this attribute is a progression, so get a constant matrix of all zeros except for the i-th column
            all_rules = []
            for delta in [1, -1, 2, -2]:
                c1 = np.zeros((3, data.shape[2]))
                c2 = c1.copy()
                c1[0, i] = 1
                c1[1, i] = -1
                c1[0, data.shape[2] - 1] = delta
                c2[1, i] = 1
                c2[2, i] = -1
                c2[0, data.shape[2] - 1] = delta
                all_rules.append(rule_and(list(rule_matrix(c) for c in [c1, c2])))
            return any(rule(data) for rule in all_rules)
        if len(np.array(data).shape) == 3:
            grid = data
            # we have a 2d matrix of embeddings
            num_rows = np.array(grid).shape[0]
            for delta in [1, -1, 2, -2]:
                if all(grid[row_i][0][i] + delta == grid[row_i][1][i] and grid[row_i][1][i] + delta == grid[row_i][2][i] for row_i in range(num_rows)):
                    return True
            return False
        row = data
        return row[0][i] == row[1][i] + 1 and row[1][i] == row[2][i] + 1
    return f
def first_index(sequence):
    for index, item in enumerate(sequence):
        if item:
            return index
    return None

def arithmetic(i, is_bitwise=False, **kwargs):
    def f(data):
        if use_composed_rules:
            data = np.array(data)
            # add a constant 1 to the end of data in its 3rd dimension
            data = np.concatenate([data, np.ones((data.shape[0], data.shape[1], 1))], axis=2)
            all_rules = []
            if is_bitwise:
                equations = []
                equations.append(lambda c1, c2, c3: int(c1) & ~int(c2) == int(c3))
                equations.append(lambda c1, c2, c3: int(c1) | int(c2) == int(c3))
                return any(all(equation(data[row_i][0][i], data[row_i][1][i], data[row_i][2][i]) for row_i in range(data.shape[0])) for equation in equations)
            else:
                for sign in [1, -1]:
                    for constant in [0, 1]:
                        c = np.zeros((3, data.shape[2]))
                        c[0, i] = 1
                        c[1, i] = sign
                        c[2, i] = -1
                        c[0, data.shape[2] - 1] = constant * sign
                        all_rules.append(rule_matrix(c))
                return any(rule(data) for rule in all_rules)
        if len(np.array(data).shape) == 3:
            grid = data
            num_rows = np.array(grid).shape[0]
            equations = []
            if is_bitwise:
                equations.append(lambda c1, c2, c3: c1 & ~c2 == c3)
                equations.append(lambda c1, c2, c3: c1 | c2 == c3)
            else:
                equations += [lambda c1, c2, c3, sign=sign, constant=constant: c1 + sign * (c2 + constant) == c3 for sign in [1, -1] for constant in [0, 1]]
            
            return any(all(equation(grid[row_i][0][i], grid[row_i][1][i], grid[row_i][2][i]) for row_i in range(num_rows)) for equation in equations)
        return False
    return f
def distribute_three(i, **kwargs):
    def f(data):
        if use_composed_rules:
            num_rows = len(data)
            must_contain = rule_set_of_values([data[0][col][i] for col in range(3)])
            return all(must_contain([data[row_i][col][i] for col in range(3)]) for row_i in range(num_rows))
        if len(np.array(data).shape) == 3:
            grid = data
            num_rows = np.array(grid).shape[0]
            # we have a 2d matrix of embeddings
            values = [set([grid[row_i][col_i][i] for col_i in range(3)]) for row_i in range(num_rows)]
            if len(values[0]) != 3:
                return False
            return all(values[row_i] == values[0] for row_i in range(num_rows))
        row = data
        return row[0][i] != row[1][i] and row[1][i] != row[2][i] and row[0][i] != row[2][i]
    return f
def noise(i, **kwargs):
    def f(row):
        return True
    return f