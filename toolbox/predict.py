from rules import *
def get_valid_rules(grid, attribute, embedding_names, single=True, allow_noise=True, debug=False):
    res = []
    for rule_i, rule in enumerate(rule_order):
        if rule == noise and not allow_noise:
            continue
        if rule(attribute, is_bitwise=('BW' in embedding_names[attribute]))([grid[0], grid[1]]):
            if debug:
                print(f'Attribute {attribute} {embedding_names[attribute]} follows rule {rule_i} {rule.__name__} {"SINGLE" if single else "MULTIPLE"}')
            if single:
                return rule_i, rule
            res.append(rule)
    return res
def predict(embeddings, embedding_names, deep_search=True, debug=False, prob=None):
    counts = [0 for _ in rule_order]
    y_rules = [0 for _ in rule_order]
    grid = [[0, 1, 2], [3, 4, 5], [6, 7, None]]
    grid = [[embeddings[col_index] if col_index is not None else None for col_index in row] for row in grid]
    if debug:
        print("\n".join([str(row) for row in grid]))
    final_rules = {}
    for attribute in range(2, len(embeddings[0])): # shape, size, color, angle
        rule_i, rule = get_valid_rules(grid, attribute, embedding_names, single=True, allow_noise=True, debug=debug)
        final_rules[attribute] = rule
        counts[rule_i] += 1
    # Guess which item is correct
    my_guess = None
    scores = [0 for _ in range(8)]
    for guess in range(8):
        y_rules = [0 for _ in rule_order]
        guess_index = 8 + guess
        guess_embedding = embeddings[guess_index]
        grid[2][2] = guess_embedding
        flag = True
        for attribute, rule in final_rules.items():
            if not rule(attribute, is_bitwise=('BW' in embedding_names[attribute]))(grid):
                if flag and debug:
                    print(f'Guess {guess} failed rule {rule.__name__} for attribute {attribute} {embedding_names[attribute]}')
                flag = False
            else:
                y_rules[rule_order.index(rule)] += 1
                scores[guess] += 1
        if flag:
            my_guess = guess
            if debug:
                print(f'My guess is {guess} with embedding {guess_embedding}')
    # print("Correct answer is", answer, "which is", embeddings[8 + answer])
    if my_guess is None:
        scores = [0 for _ in range(8)]
        y_rules = [0 for _ in rule_order]
        y_rules_for_each = [y_rules.copy() for _ in range(8)]
        if deep_search:
            scores = [0 for _ in range(8)]
            # Try all possible rules that are valid
            rules = {}
            for attribute in range(2, len(embeddings[0])):
                rules[attribute] = get_valid_rules(grid, attribute, embedding_names, single=False, allow_noise=True, debug=debug)
            for guess in range(8):
                guess_index = 8 + guess
                guess_embedding = embeddings[guess_index]
                grid[2][2] = guess_embedding
                flag = True
                for attribute in range(2, len(embeddings[0])):
                    # get the first valid rule that rule(attribute) returns True
                    first_valid_rule = next((rule for rule in rules[attribute] if rule(attribute, is_bitwise=('BW' in embedding_names[attribute]))(grid)), None)
                    if first_valid_rule is None:
                        flag = False
                        if debug:
                            print(f'Guess {guess} failed rule {rule.__name__} for attribute {attribute} {embedding_names[attribute]}')
                        break
                    else:
                        if first_valid_rule != noise:
                            scores[guess] += 1
                        y_rules_for_each[guess][rule_order.index(first_valid_rule)] += 1
                if flag:
                    # my_guess = np.argmax(scores)
                    if debug:
                        print(f'Guess {guess} is valid with embedding {guess_embedding}')
        my_guess = np.argmax(scores)
        y_rules = y_rules_for_each[my_guess]
        if debug:
            print("No answer found, guessing", my_guess, " from scores", scores)
    return my_guess, counts, y_rules

def assert_properties(shapes, embeddings, answer, debug=False):
    for shape_index, shape in enumerate(shapes):
        # Check if any of the dimensions are constant
        for attribute_index in range(len(shape[0])):
            if all([shape[0][attribute_index] == shape[grid_index][attribute_index] for grid_index in range(1, len(shape))]):
                # Make sure the answer is the same
                if debug:
                    print(f'Attribute {attribute_index} is constant for shape {shape_index} and is {shape[0][attribute_index]}')
                    print(f'Answer is {embeddings[8 + answer]}')
                assert shape[0][attribute_index] == embeddings[8 + answer][shape_index][attribute_index], f'Attribute {attribute_index} is constant but answer is not the same'
# There are a lot of rule combinations that work. Which ones generate the answer we are looking for? I dont know!

def predict_shapes(shapes, embeddings, debug=False):
    return 0
    # create some combination rules that are composed together.
    counts = [0 for _ in rule_order]
    y_rules = [0 for _ in rule_order]
    grid = [[0, 1, 2], [3, 4, 5], [6, 7, None]]
    grid = [[embeddings[col_index] if col_index is not None else None for col_index in row] for row in grid]
    if debug:
        print("\n".join([str(row) for row in grid]))
    final_rules = {}
    for attribute in range(2, len(embeddings[0])): # shape, size, color, angle
        rule_i, rule = get_valid_rules(grid, attribute, embedding_names, single=True, allow_noise=True, debug=debug)
        final_rules[attribute] = rule
        counts[rule_i] += 1
    # Guess which item is correct
    my_guess = None
    scores = [0 for _ in range(8)]
    for guess in range(8):
        y_rules = [0 for _ in rule_order]
        guess_index = 8 + guess
        guess_embedding = embeddings[guess_index]
        grid[2][2] = guess_embedding
        flag = True
        for attribute, rule in final_rules.items():
            if not rule(attribute, is_bitwise=('BW' in embedding_names[attribute]))(grid):
                if flag and debug:
                    print(f'Guess {guess} failed rule {rule.__name__} for attribute {attribute} {embedding_names[attribute]}')
                flag = False
            else:
                y_rules[rule_order.index(rule)] += 1
                scores[guess] += 1
        if flag:
            my_guess = guess
            if debug:
                print(f'My guess is {guess} with embedding {guess_embedding}')
    # print("Correct answer is", answer, "which is", embeddings[8 + answer])
    if my_guess is None:
        scores = [0 for _ in range(8)]
        y_rules = [0 for _ in rule_order]
        y_rules_for_each = [y_rules.copy() for _ in range(8)]
        if deep_search:
            scores = [0 for _ in range(8)]
            # Try all possible rules that are valid
            rules = {}
            for attribute in range(2, len(embeddings[0])):
                rules[attribute] = get_valid_rules(grid, attribute, embedding_names, single=False, allow_noise=True, debug=debug)
            for guess in range(8):
                guess_index = 8 + guess
                guess_embedding = embeddings[guess_index]
                grid[2][2] = guess_embedding
                flag = True
                for attribute in range(2, len(embeddings[0])):
                    # get the first valid rule that rule(attribute) returns True
                    first_valid_rule = next((rule for rule in rules[attribute] if rule(attribute, is_bitwise=('BW' in embedding_names[attribute]))(grid)), None)
                    if first_valid_rule is None:
                        flag = False
                        if debug:
                            print(f'Guess {guess} failed rule {rule.__name__} for attribute {attribute} {embedding_names[attribute]}')
                        break
                    else:
                        if first_valid_rule != noise:
                            scores[guess] += 1
                        y_rules_for_each[guess][rule_order.index(first_valid_rule)] += 1
                if flag:
                    # my_guess = np.argmax(scores)
                    if debug:
                        print(f'Guess {guess} is valid with embedding {guess_embedding}')
        my_guess = np.argmax(scores)
        y_rules = y_rules_for_each[my_guess]
        if debug:
            print("No answer found, guessing", my_guess, " from scores", scores)
    return my_guess, counts, y_rules
