import random
import numpy as np
import itertools
indices_used = list(itertools.product(itertools.permutations(range(4), 2), itertools.permutations(range(3), 2), itertools.permutations(range(2), 2)))
operations_used = list(itertools.product(range(4), repeat=3)) # 4 operations, using 3 at a time
operations = [
    lambda x, y: x + y,
    lambda x, y: x - y,
    lambda x, y: x * y,
    lambda x, y: x / y,
]
operation_names = ["+", "-", "*", "/"]
combos = list(itertools.product(indices_used, operations_used))
def get_dataset(min_val=1, max_val=12, target=24, filename=None, create_data=False, tqdm=lambda x: x):
    filename = f"questions{max_val}_{target}.txt" if filename is None else filename
    create_data = False
    def get_questions():
        res = []
        for a in tqdm(range(min_val, max_val + 1)):
            for b in range(a, max_val + 1):
                for c in range(b, max_val + 1):
                    for d in range(c, max_val + 1):
                        for index_order, operation_order in itertools.product(indices_used, operations_used):
                            numbers = [a, b, c, d]
                            valid = True
                            for (index1, index2), operation_index in zip(index_order, operation_order):
                                val1 = numbers[index1]
                                val2 = numbers[index2]
                                numbers.pop(max(index1, index2))
                                numbers.pop(min(index1, index2))
                                if operation_index == 3 and val2 == 0:
                                    valid = False
                                    break
                                result = operations[operation_index](val1, val2)
                                if result % 1 != 0 or result < 0:
                                    valid = False
                                    break
                                numbers.append(result)
                            if valid and numbers[0] % 1 == 0 and numbers[0] == 24:
                                res.append((a, b, c, d, 24))
                                break
        return res
    if create_data:
        questions = get_questions()
        with open(filename, "w") as file:
            file.write("\n".join([f"{' '.join([str(item) for item in items])}" for items in questions]))
    else:
        with open(filename, "r") as file:
            questions = file.read().split("\n")
            questions = [list(map(int, question.split())) for question in questions]
    mode = ['random_choice', 'random_list', 'ordered'][0]
    def solve(question):
        # pair up two numbers 1-4, then 1-3, then 1-2.
        # For each pair, we can add, subtract, multiply or divide.
        target = question[-1]
        choices = list(question[:-1]).copy()
        # shuffle it
        if mode == 'random_list':
            random.shuffle(choices)
        i = 0
        while mode == 'random_choice' or i < len(combos):
            index_order, operation_order = random.choice(combos) if mode == 'random_choice' else combos[i]
            i += 1
            numbers = choices.copy()
            valid = True
            for (index1, index2), operation_index in zip(index_order, operation_order):
                val1 = numbers[index1]
                val2 = numbers[index2]
                numbers.pop(max(index1, index2))
                numbers.pop(min(index1, index2))
                if operation_index == 3 and val2 == 0:
                    valid = False
                    break
                numbers.append(operations[operation_index](val1, val2))
            if valid and numbers[0] == target:
                return True, index_order, operation_order, i
        return False, None, None, -1
    def print_solution(numbers, index_order, operation_order, answer):
        terms = [str(x) for x in numbers]
        for (index1, index2), operation_index in zip(index_order, operation_order):
            term1 = terms[index1]
            term2 = terms[index2]
            terms.pop(max(index1, index2))
            terms.pop(min(index1, index2))
            new_term = f"({term1} {operation_names[operation_index]} {term2})"
            terms.append(new_term)
        print(" ".join(terms), "=", answer)
    debug = False
    avg_count = 0
    n_entries = 0
    solutions = []
    min_count = 100000
    max_count = 0
    for question in tqdm(questions):
        valid, index_order, operation_order, i = solve(question)
        if not valid:
            print("Failed to solve", question)
            break
        else:
            min_count = min(min_count, i)
            max_count = max(max_count, i)
            avg_count += i
            n_entries += 1
            solutions.append([question[:-1], question[-1], index_order, operation_order])
            if debug:
                print("input", question, "indices", index_order, "operations", operation_order, "count", i)
                print_solution(question[0], index_order, operation_order, question[1])
                print()
    return questions, solutions

def augment_data(X, y, n):
    X_new = []
    y_new = []
    for i in range(X.shape[0] * n):
        random_index = random.randint(0, X.shape[0]-1)
        X_new.append(np.random.permutation(X[random_index]))
        y_new.append(y[random_index])
    return np.array(X_new), np.array(y_new)