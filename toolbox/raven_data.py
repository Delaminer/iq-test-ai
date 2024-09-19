import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import matplotlib.patches as patches
import ast


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