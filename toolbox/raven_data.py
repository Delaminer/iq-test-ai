import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import matplotlib.patches as patches
import ast
import random, math
import cv2

def xml_to_dict(element):
    result = {**element.attrib} if element.attrib else {}
    if len(element):
        # Create a dictionary for the child elements
        children = []
        for child in element:
            # Recursively convert each child and update the dictionary
            children.append(xml_to_dict(child))
        result[element.tag] = children
    else:
        # If no children, just store the text content
        result[element.tag] = element.text.strip() if element.text else None
    return result

def parse_xml_to_dict(file):
    tree = ET.parse(file)
    root = tree.getroot()
    return xml_to_dict(root)

def display_problem(images, answer=None, my_display_boxes=None):
    first_nine = images[:9].copy()
    first_nine[8] = np.zeros(images[8].shape) if answer is None else images[8 + answer].copy()
    fig, ax = plt.subplots(3, 3)
    for a in ax.ravel():
        a.set_aspect('equal')
    fig.set_facecolor('lightgray')
    box_index = 0
    for i in range(3):
        for j in range(3):
            ax[i, j].imshow(first_nine[i * 3 + j], cmap='gray')

            my_boxes = my_display_boxes[box_index] if my_display_boxes is not None else []
            box_index += 1
            for inner_box_index, box in enumerate(my_boxes):
                height, width = first_nine[i * 3 + j].shape
                orig_y, orig_x, orig_height, orig_width = box # scaled from 0 to 1
                # orig_x, orig_y, orig_width, orig_height = [0.5, 0.5, 0.1, 0.1] # scaled from 0 to 1
                box_width, box_height = orig_width * width, orig_height * height
                box_x = orig_x * width - box_width / 2
                box_y = orig_y * height - box_height / 2
                # Add centered red bounding box
                rect = patches.Rectangle((box_x, box_y), box_width, box_height,
                                        linewidth=2, edgecolor=['red', 'blue'][inner_box_index % 2], facecolor='none')
                ax[i, j].add_patch(rect)

            ax[i, j].axis('off')
    plt.show()
    print(f"Correct answer: {answer}")
    # Show the 8 possible answers
    fig, ax = plt.subplots(2, 4)
    for a in ax.ravel():
        a.set_aspect('equal')
    fig.set_facecolor('lightgray')
    for i in range(2):
        for j in range(4):
            ax[i, j].imshow(images[8 + i * 4 + j], cmap='gray')
            ax[i, j].axis('off')
    plt.show()

def load_question(filebase, display=False, debug=False):
    # Read the .xml file
    data = parse_xml_to_dict(f'{filebase}.xml')
    embeddings = []
    embedding_names = None
    all_bbox = []
    for i in range(16):
        # Get all possible positions
        components = data['Data'][0]['Panels'][i]['Panel'][0]['Struct']
        for j in range(len(components)):
            entities = components[j]['Component'][0]['Layout']
            for entity in entities:
                bbox = entity['bbox'] # keep as a string since it can be used as a key
                all_bbox.append(bbox)
                if debug:
                    print(bbox)
        if debug:
            print("done")
    # Map each unique bounding box to an index
    bbox_to_index = {}
    for bbox in sorted(all_bbox):
        if bbox not in bbox_to_index:
            bbox_to_index[bbox] = len(bbox_to_index)
    if debug:
        print(bbox_to_index)
    display_boxes = []
    for i in range(16):
        my_display_boxes = []
        row = i // 3
        col = i % 3
        if i > 8:
            row, col = 2, 2
        embedding = [row, col]
        embedding_names = ['Row', 'Col']
        components = data['Data'][0]['Panels'][i]['Panel'][0]['Struct']
        for j in range(len(components)):
            obj = components[j]['Component'][0]['Layout'][0]
            base_attributes = ['Type', 'Size', 'Color', 'Angle']
            embedding += [int(obj[attribute]) for attribute in base_attributes]
            embedding_names += base_attributes
            entities = components[j]['Component'][0]['Layout']
            position_indices = {}
            for pos in ast.literal_eval(components[j]['Component'][0]['Position']):
                position_indices[str(pos)] = len(position_indices)
            position_encoding = 0 # mark each bit for each bbox present
            for entity in entities:
                bbox = entity['bbox']
                my_display_boxes.append(ast.literal_eval(entity['bbox']))
                my_display_boxes.append(ast.literal_eval(entity['real_bbox']))
                bbox_index = position_indices[bbox]
                position_encoding |= 1 << bbox_index
            embedding.append(position_encoding)
            embedding_names.append('BWPosition')
            
            embedding.append(len(entities))
            embedding_names.append('Number')
        embeddings.append(embedding)
        display_boxes.append(my_display_boxes)
    
    data_npz = np.load(f'{filebase}.npz')
    if display:
        display_problem(data_npz['image'], data_npz['target'], display_boxes if debug else None)
    answer = data_npz['target']

    return embeddings, embedding_names, answer


def find_polygon_sides(bitwise_image):
    # Convert the bitwise image (0s and 1s) to uint8 type for OpenCV processing
    image_uint8 = (bitwise_image * 255).astype(np.uint8)
    
    # Find contours in the image
    contours, _ = cv2.findContours(image_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return "No shape found"
    
    # Assuming the largest contour is the shape we're looking for
    contour = max(contours, key=cv2.contourArea)

    # Calculate the perimeter and area of the contour
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    
    # Calculate circularity
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    
    # If circularity is close to 1, classify it as a circle
    if circularity > 0.85:  # Adjust the threshold as necessary
        return 1  # Circle
    
    # Approximate the contour to a polygon
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # The number of sides is the number of vertices of the approximated polygon
    num_sides = len(approx)
    
    return num_sides

def find_polygon_rotation_angle(bitwise_image):
    # Convert the bitwise image (0s and 1s) to uint8 type for OpenCV processing
    image_uint8 = (bitwise_image * 255).astype(np.uint8)
    
    # Find contours in the image
    contours, _ = cv2.findContours(image_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return "No shape found"
    
    # Assuming the largest contour is the shape we're looking for
    contour = max(contours, key=cv2.contourArea)
    
    # Get the minimum area rectangle around the shape
    rect = cv2.minAreaRect(contour)
    
    # The angle of rotation is stored in rect[2]
    angle = rect[2]
    
    # Adjust angle to make it consistent (if needed, depending on OpenCV version)
    if angle < -45:
        angle += 90
    
    return angle, rect

def detect_img_features(img, masks, index, shape=(160, 160)):
    img_mask = masks[index]
    # plt.imshow(img_mask)
    # plt.show()
    # plt.imshow(img)
    # plt.show()
    # plt.imshow(img * img_mask)
    # plt.show()
    size = np.sum(img_mask)
    pos = list(np.mean(np.where(img_mask), axis=1))
    sides = find_polygon_sides(img_mask)
    # For color, take the image mask and remove smaller shapes that lie within this shape
    my_bbox = cv2.boundingRect(img_mask.astype(np.uint8))
    img_mask_removed = img_mask.copy()
    for i in range(len(masks)):
        if i == index:
            continue
        # it is smaller if the bounding box is contained within the my bounding box
        other_mask_bbox = cv2.boundingRect(masks[i].astype(np.uint8))
        other_mask_is_smaller = my_bbox[0] <= other_mask_bbox[0] and my_bbox[1] <= other_mask_bbox[1] and my_bbox[0] + my_bbox[2] >= other_mask_bbox[0] + other_mask_bbox[2] and my_bbox[1] + my_bbox[3] >= other_mask_bbox[1] + other_mask_bbox[3]
        if other_mask_is_smaller:
            img_mask_removed = np.logical_and(img_mask_removed, np.logical_not(masks[i]))
    
    color = np.mean(img[img_mask_removed.astype(bool)])
    angle, rect = find_polygon_rotation_angle(img_mask)
    # plot rotated rectangle
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    im2 = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    cv2.drawContours(im2, [box], 0, (255, 255, 255), 2)
    # add image mask to im2 as red
    im2[img_mask.astype(bool)] = [255, 0, 0]
    # plt.imshow(im2)
    # plt.show()
    # print(angle)
    # print("Size", size, "Pos", pos, "Sides", sides, "Color", color, "Angle", angle)
    return size, pos, sides, color, angle


def load_question3(filebase, parse_mask=True, display=False, debug=False):
    # Read the .xml file
    data = parse_xml_to_dict(f'{filebase}.xml')
    data_npz = np.load(f'{filebase}.npz')
    embeddings = []
    embedding_names = None
    all_bbox = []
    for i in range(16):
        # Get all possible positions
        components = data['Data'][0]['Panels'][i]['Panel'][0]['Struct']
        for j in range(len(components)):
            entities = components[j]['Component'][0]['Layout']
            for entity in entities:
                bbox = entity['bbox'] # keep as a string since it can be used as a key
                all_bbox.append(bbox)
                if debug:
                    print(bbox)
        if debug:
            print("done")
    # Map each unique bounding box to an index
    bbox_to_index = {}
    for bbox in sorted(all_bbox):
        if bbox not in bbox_to_index:
            bbox_to_index[bbox] = len(bbox_to_index)
    if debug:
        print(bbox_to_index)
    display_boxes = []
    for i in range(16):
        my_display_boxes = []
        embedding = []
        embedding_names = []
        components = data['Data'][0]['Panels'][i]['Panel'][0]['Struct']
        all_entity_masks = []
        for j in range(len(components)):
            entities = components[j]['Component'][0]['Layout']
            all_entity_masks += [rle_decode(entity['mask'], (160, 160), fill='concave') for entity in entities]
        all_entity_index = 0
        for j in range(len(components)):
            entities = components[j]['Component'][0]['Layout']
            for entity_index, entity in enumerate(entities):
                if parse_mask:
                    size, pos, sides, color, angle = detect_img_features(data_npz['image'][i], all_entity_masks, all_entity_index)
                    all_entity_index += 1
                    embedding_names += ['Size', 'Position', 'Sides', 'Color', 'Angle']
                    embedding.append([size, pos, sides, color, angle])
                else:
                    base_attributes = ['Type', 'Size', 'Color', 'Angle']

                    bbox = ast.literal_eval(entity['bbox'])
                    position = [bbox[0], bbox[1]]
                    actual_size = [bbox[2], bbox[3]]

                    entity_embedding = [int(entity[attribute]) for attribute in base_attributes] + [position, actual_size]
                    embedding_names += base_attributes + ['Position', 'ActualSize']
                    embedding.append(entity_embedding)
            # embedding.append(len(entities))
            # embedding_names.append('Number')
        embeddings.append(embedding)
        display_boxes.append(my_display_boxes)
    
    if display:
        display_problem(data_npz['image'], data_npz['target'], display_boxes if debug else None)
    answer = data_npz['target']

    return embeddings, embedding_names, answer

def similarity(embedding1, embedding2):
    score = 1
    assert len(embedding1) == len(embedding2)
    for i in range(len(embedding1)):
        if type(embedding1[i]) == list: # 2d coord or size
            # manhattan distance
            assert len(embedding1[i]) == len(embedding2[i])
            diff_value = sum([abs(embedding1[i][j] - embedding2[i][j]) for j in range(len(embedding1[i]))])
        else:
            diff_value = abs(embedding1[i] - embedding2[i])
        score *= math.sqrt(diff_value + 1)
    return score

def entity_id(cell_index, entity_index):
    return f"{cell_index}_{entity_index}"

def close_values(value1, value2, threshold=0.1):
    if value1 == value2:
        return True
    if type(value1) == list:
        return all([(abs(value1[i] - value2[i]) / max(value1[i], value2[i])) < threshold for i in range(len(value1))])
    return (abs(value1 - value2) / max(abs(value1), abs(value2))) < threshold

# next define_shapes that uses the fact that constant number of entities in each cell means each entity is corresponds to a shape
def define_shapes3(embeddings, embedding_names, debug=False):
    constant_number_of_entities = all([len(embedding) == len(embeddings[0]) for embedding in embeddings])
    shapes = []
    # find entities in each cell that have the same range of values for a specific attribute
    for entity_index, entity in enumerate(embeddings[0]):
        stop_search = False # if we can't find any attributes that are the same across all cells, then stop searching
        valid_attributes = [True for _ in range(len(entity))] # which attributes are "constant" across all cells
        use_which_other_entities = [[] for _ in range(len(entity))] # which other entities from other cells should be used because they have the same attribute
        # for other_cell_index, other_embedding in enumerate(embeddings[1:8]):
        for other_cell_index in range(1, 8):
            other_embedding = embeddings[other_cell_index]
            for attribute_index, attribute in enumerate(entity):
                if not valid_attributes[attribute_index]:
                    continue
                some_other_entity_has_same_attribute = False

                for other_entity_index, other_entity in enumerate(other_embedding):
                    if close_values(attribute, other_entity[attribute_index]):
                        some_other_entity_has_same_attribute = True
                        use_which_other_entities[attribute_index].append((other_cell_index, other_entity_index))
                if not some_other_entity_has_same_attribute:
                    valid_attributes[attribute_index] = False
                    if not any(valid_attributes):
                        stop_search = True
                        break
            if stop_search:
                break
        if stop_search:
            if debug:
                print("Couldnt find any attributes that are the same across all cells for entity", entity_index, entity)
        else:
            if debug:
                print("Found attributes that are the same across all cells for entity", entity_index, entity)
                print(valid_attributes)
            for attribute_index, (attribute, use_entities) in enumerate(zip(entity, use_which_other_entities)):
                if valid_attributes[attribute_index] and len(use_entities) == 7:
                    if debug:
                        print("Attribute", attribute_index, "is the same across all cells")
                        print("Use entities", use_entities)
                    # Make a shape, which is a list of the embeddings across the 8 cells for this shape
                    shapes.append([embeddings[cell_index][entity_index] for cell_index, entity_index in [(0, entity_index)] + use_entities])
                    break
    return shapes

# Better define_shapes that uses the fact that everyone agrees on the same attribute values
def define_shapes(embeddings, embedding_names, debug=False):
    shapes = []
    # find entities in each cell that have the same range of values for a specific attribute
    for entity_index, entity in enumerate(embeddings[0]):
        stop_search = False # if we can't find any attributes that are the same across all cells, then stop searching
        valid_attributes = [True for _ in range(len(entity))] # which attributes are "constant" across all cells
        use_which_other_entities = [[] for _ in range(len(entity))] # which other entities from other cells should be used because they have the same attribute
        # for other_cell_index, other_embedding in enumerate(embeddings[1:8]):
        for other_cell_index in range(1, 8):
            other_embedding = embeddings[other_cell_index]
            for attribute_index, attribute in enumerate(entity):
                if not valid_attributes[attribute_index]:
                    continue
                some_other_entity_has_same_attribute = False

                for other_entity_index, other_entity in enumerate(other_embedding):
                    if close_values(attribute, other_entity[attribute_index]):
                        some_other_entity_has_same_attribute = True
                        use_which_other_entities[attribute_index].append((other_cell_index, other_entity_index))
                if not some_other_entity_has_same_attribute:
                    valid_attributes[attribute_index] = False
                    if not any(valid_attributes):
                        stop_search = True
                        break
            if stop_search:
                break
        if stop_search:
            if debug:
                print("Couldnt find any attributes that are the same across all cells for entity", entity_index, entity)
        else:
            if debug:
                print("Found attributes that are the same across all cells for entity", entity_index, entity)
                print(valid_attributes)
            for attribute_index, (attribute, use_entities) in enumerate(zip(entity, use_which_other_entities)):
                if valid_attributes[attribute_index] and len(use_entities) == 7:
                    if debug:
                        print("Attribute", attribute_index, "is the same across all cells")
                        print("Use entities", use_entities)
                    # Make a shape, which is a list of the embeddings across the 8 cells for this shape
                    shapes.append([embeddings[cell_index][entity_index] for cell_index, entity_index in [(0, entity_index)] + use_entities])
                    break
    return shapes

# Uses shapes where everyone agrees they prefer each other. Do not use
def define_shapes2(embeddings, embedding_names):
    shapes = []
    all_scores = [] # cell_index -> entity_index -> other_cell_index -> other_entity_index -> score
    preferences = [] # cell_index -> entity_index -> other_cell_index -> preference of which entity to match with
    # Group similar shapes together
    for cell_index, embedding in enumerate(embeddings[:8]):
        cell_data = []
        cell_preferences = []
        for entity_index, entity in enumerate(embedding):
            entity_data = []
            entity_preferences = []
            for other_cell_index, other_embedding in enumerate(embeddings[:8]):
                scores = [similarity(entity, other_entity) for other_entity in other_embedding] if other_cell_index != cell_index else [-1] * len(other_embedding)
                entity_data.append(scores)
                entity_preferences.append(np.argmin(scores) if other_cell_index != cell_index else entity_index)
            cell_data.append(entity_data)
            cell_preferences.append(entity_preferences)
        print(cell_data)
        all_scores.append(cell_data)
        preferences.append(cell_preferences)
    print(all_scores)
    print(preferences)

    # if everyone prefers each other, then we can match them as one shape
    for entity_index, entity in enumerate(embeddings[0]):
        they_also_prefer_me = True
        for other_cell_index, other_embedding in enumerate(embeddings[:8]):
            if preferences[0][entity_index][other_cell_index] != entity_index:
                they_also_prefer_me = False
                break
        if they_also_prefer_me:
            print(entity_index)
            # Make a shape, which is a list of the embeddings across the 8 cells for this shape
            shapes.append([embeddings[cell_index][preferences[0][entity_index][cell_index]] for cell_index in range(8)])
    return shapes

def rle_decode(mask_rle, shape, fill='rle'):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    fill: 'rle' - fill the mask with 1s, 'edge' - fill the edges of the mask with 1s, 'concave' - fill the concave shape with 1s (based on what edges are found)
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle[1:-1].split(",")
    if len(s) % 2 != 0:
        # print("watch out invalid mask rle pattern so we will add a 1 to the end")
        s.append('1')
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    concave_bounds = np.zeros((shape[0], 2), dtype=np.uint8) # Store the left and right bounds of the concave shape for each row
    concave_bounds[:, 0] = shape[1] # Initialize to the maximum value
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        if fill == 'rle':
            img[lo:hi] = 1
        if fill == 'edge':
            img[lo] = 1
            img[hi - 1] = 1
        if fill == 'concave':
            concave_bounds[lo // shape[1], 0] = min(lo % shape[1], concave_bounds[lo // shape[1], 0])
            concave_bounds[lo // shape[1], 1] = max(hi % shape[1], concave_bounds[lo // shape[1], 1])
    img = img.reshape(shape)
    if fill == 'concave':
        for row in range(img.shape[0]):
            if concave_bounds[row, 1] > 0:
                leftmost = concave_bounds[row, 0]
                rightmost = concave_bounds[row, 1]
                img[row, leftmost:rightmost + 1] = 1
    return img

# for embeddings.ipynb, this prints out masks made by RLE decode and isnt useful
def load_question2(filebase, display=False, debug=False):
    # Read the .xml file
    data = parse_xml_to_dict(f'{filebase}.xml')
    embeddings = []
    embedding_names = None
    all_bbox = []
    all_masks = []
    for i in range(16):
        # Get all possible positions
        components = data['Data'][0]['Panels'][i]['Panel'][0]['Struct']
        masks = []
        for j in range(len(components)):
            entities = components[j]['Component'][0]['Layout']
            for entity in entities:
                bbox = entity['bbox'] # keep as a string since it can be used as a key
                all_bbox.append(bbox)
                masks.append(rle_decode(entity['mask'], (160, 160)))
                if debug:
                    print(bbox)
        all_masks.append(masks)
    data_npz = np.load(f'{filebase}.npz')
    display_problem(data_npz['image'], data_npz['target'], None)
    for mask_group in all_masks:
        merged_masks = np.zeros((160, 160))
        for mask in mask_group:
            merged_masks = np.logical_or(merged_masks, mask)
        plt.imshow(merged_masks)
        plt.show()
    return masks
    # Map each unique bounding box to an index
    bbox_to_index = {}
    for bbox in sorted(all_bbox):
        if bbox not in bbox_to_index:
            bbox_to_index[bbox] = len(bbox_to_index)
    if debug:
        print(bbox_to_index)
    display_boxes = []
    for i in range(16):
        my_display_boxes = []
        row = i // 3
        col = i % 3
        if i > 8:
            row, col = 2, 2
        embedding = [row, col]
        embedding_names = ['Row', 'Col']
        components = data['Data'][0]['Panels'][i]['Panel'][0]['Struct']
        for j in range(len(components)):
            obj = components[j]['Component'][0]['Layout'][0]
            base_attributes = ['Type', 'Size', 'Color', 'Angle']
            embedding += [int(obj[attribute]) for attribute in base_attributes]
            embedding_names += base_attributes
            entities = components[j]['Component'][0]['Layout']
            position_indices = {}
            for pos in ast.literal_eval(components[j]['Component'][0]['Position']):
                position_indices[str(pos)] = len(position_indices)
            position_encoding = 0 # mark each bit for each bbox present
            for entity in entities:
                bbox = entity['bbox']
                my_display_boxes.append(ast.literal_eval(entity['bbox']))
                my_display_boxes.append(ast.literal_eval(entity['real_bbox']))
                bbox_index = position_indices[bbox]
                position_encoding |= 1 << bbox_index
            embedding.append(position_encoding)
            embedding_names.append('BWPosition')
            
            embedding.append(len(entities))
            embedding_names.append('Number')
        embeddings.append(embedding)
        display_boxes.append(my_display_boxes)
    
    data_npz = np.load(f'{filebase}.npz')
    if display:
        display_problem(data_npz['image'], data_npz['target'], display_boxes if debug else None)
    answer = data_npz['target']

    return embeddings, embedding_names, answer