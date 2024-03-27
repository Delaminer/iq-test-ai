import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance as dist
import math
from scipy import ndimage
from skimage import measure
import numpy as np

def draw_contours(image, contours, same_color=False):
    # Rainbow colors in BGR format
    rainbow_colors = [
        (0, 0, 255),     # Red
        (0, 165, 255),   # Orange
        (0, 255, 255),   # Yellow
        (0, 255, 0),     # Green
        (255, 0, 0),     # Blue
        (130, 0, 75),    # Indigo
        (238, 130, 238)  # Violet
    ]

    image_with_contours = image.copy()
    
    # Loop through contours and colors
    for i, contour in enumerate(contours):
        color = rainbow_colors[i % len(rainbow_colors)] if not same_color else (0, 255, 0)
        cv2.drawContours(image_with_contours, [contour], -1, color, 2)

    return image_with_contours

def show_image(image):
    if len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def show_images_grid(images):
    # Display the cropped images for visual verification
    grid_size = np.sqrt(len(images))
    # round up to the nearest integer
    grid_size = math.ceil(grid_size)
    # if this does not require all the rows allocated, remove some
    cols = grid_size
    rows = math.ceil(len(images) / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(10, 10))
    for i, ax in enumerate(axs.flat):
        if i < len(images):
            if len(images[i].shape) == 2:
                ax.imshow(images[i], cmap='gray')
            else:
                ax.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def get_overlap(image1, image2):
    assert image1.shape == image2.shape
    a = np.zeros((image1.shape[0], image1.shape[1], 3), dtype=np.uint8)
    a[np.where(image1 > 0)] = (0, 255, 0)
    b = np.zeros((image1.shape[0], image1.shape[1], 3), dtype=np.uint8)
    b[np.where(image2 > 0)] = (255, 0, 0)
    return (a + b)

answers = ['A','E','F','F','D','E','E','C','D','D','A','A','B','A','F',
           'B','D','C','A','A','E','E','F','F','E','F','A','A','C','E','D','?','A?','F?','D']

def get_iq_question_images(iq_number, debug=False):
    # Convert number (1-10) to file path "iq_images/iq{i}.png" but with a leading zero if i < 10
    file_path = f"iq_images/iq{str(iq_number).zfill(2)}.png"
    image = cv2.imread(file_path)
    # invert image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if debug:
        print("All contours")
        show_image(draw_contours(image, contours))
    
    contours = [contour for contour in contours if cv2.contourArea(contour) / (image.shape[0] * image.shape[1]) < 0.95]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if debug:
        print("Contours after removing border")
        show_image(draw_contours(image, contours))

    grid_contour = contours[0]
    
    grid_x, grid_y, grid_w, grid_h = cv2.boundingRect(grid_contour)

    if debug:
        print("Grid part of the image")
        show_image(image[grid_y:grid_y+grid_h, grid_x:grid_x+grid_w])

    grid_data = []
    # Within the grid should be 9 contours that are squares of the same size (grid size / 9)
    expected_area = grid_w * grid_h / 9
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < expected_area * 0.9 or area > expected_area * 1.1:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        # ranking is to sort from left to right, top to bottom
        ranking = y * grid_w + x
        grid_data.append((ranking, gray[y:y+h, x:x+w], contour))

    grid_data = sorted(grid_data, key=lambda x: x[0])
    grid = [data[1] for data in grid_data]
    grid_contours = [data[2] for data in grid_data]
    # Remove the border from each image
    border = 3
    grid = [image[border:-border, border:-border] for image in grid]

    # Define the resize function to make each image have the same size
    desired_width = int(np.max([image.shape[1] for image in grid]))
    desired_height = int(np.max([image.shape[0] for image in grid]))
    # remove border and extra edge to 
    def resize(image):
        # Get the current dimensions of the image
        current_height, current_width = image.shape[:2]
        # Calculate the padding required to achieve the desired dimensions
        pad_height = max(0, desired_height - current_height)
        pad_width = max(0, desired_width - current_width)
        # Pad the image with black pixels
        # image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
        image = np.pad(image, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=255)
        # Slice the padded image to the desired dimensions
        remove_height = max(0, current_height - desired_height)
        remove_width = max(0, current_width - desired_width)
        remove_top = remove_height // 2
        remove_bottom = remove_height - remove_top
        remove_left = remove_width // 2
        remove_right = remove_width - remove_left

        # avoid removing more than we need to

        image = image[remove_top:current_height - remove_bottom, remove_left:current_width - remove_right]
        return image
    grid = [resize(image) for image in grid]
    # Get the average width and height of the grid images so we can crop each grid image and the choices to the same size
    # Crop the grid images to the average size
    # Invert
    grid = [cv2.bitwise_not(image) for image in grid]

    if debug:
        print("Grid contours")
        show_image(draw_contours(image, grid_contours))

        print("Resulting grid images")
        show_images_grid(grid)

    choices = []

    # get the blue lines
    blue_line_color = np.array([255, 239, 219])
    threshold = 10
    diff = np.abs(image - blue_line_color)
    only_blue_lines = np.all(diff <= threshold, axis=2)
    # make a rectangle from the area each line covers
    # first get the contours
    blue_lines, _ = cv2.findContours((only_blue_lines > 0).astype(np.uint8) * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # then get the bounding rectangles
    blue_line_rects = [cv2.boundingRect(line) for line in blue_lines]
    # sort the rectangles from left to right, then top to bottom
    blue_line_rects = sorted(blue_line_rects, key=lambda x: x[1] * image.shape[1] + x[0])
    # get the contents of each rectangle
    rects = []

    for x, y, w, h in blue_line_rects:
        # check the rectangle is a reasonable size
        if (w < 10 or h < 10):
            continue
        # check if we have already processed this rectangle
        if any([abs(x - rx) < 10 and abs(y - ry) < 10 for rx, ry, rw, rh in rects]):
            continue
        # remove the border
        x, y, w, h = x + 1, y + 1, w - 2, h - 2
        top_height = 30 if y < 600 else 35
        rects.append((x, y, w, h))
        choice = gray[y+top_height:y+h, x:x+w]
        # rescale the image (based on the y) to that of the grid
        original_height = choice.shape[0]
        new_height = grid[0].shape[0]
        scale = new_height / original_height
        choice = cv2.resize(choice, (0, 0), fx=scale, fy=scale)
        # remove excess left and right
        # original_width = choice.shape[1]
        # new_width = grid[0].shape[1]
        # # calculate the original left and right bounds of any white pixels
        # white_pixels_indices = np.argwhere(choice == 1)
        # original_left = np.min(white_pixels_indices[:, 1])
        # original_right = np.max(white_pixels_indices[:, 1])
        # center_x = (original_left + original_right) // 2
        # choice = choice[:, left:right]
        choice = resize(choice)
        choices.append(choice)
    
    # Invert
    choices = [cv2.bitwise_not(image) for image in choices]
    
    answer = answers[iq_number - 1]
    if '?' not in answer:
        # convert to index
        answer = ord(answer) - ord('A')
    else:
        answer = None

    return grid, choices, answer

def force_to_binary(image):
    # if this is a color image, convert to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # if the image is not already binary, convert it to binary
    if len(np.unique(image)) > 2:
        _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    return image

class ModelTester:
    def __init__(self):
        # each entry in dataset is a tuple of the form (grid, choices, answer)
        self.dataset = []
        for i in range(1, 36):
            grid, choices, answer = get_iq_question_images(i)
            self.dataset.append((grid, choices, answer))
    
    def test(self, get_prediction):
        correct = 0
        total = 0
        results = []
        for grid, choices, answer in self.dataset:
            if answer is not None:
                prediction = get_prediction(grid, choices)
                total += 1
                results.append(prediction == answer)
                if prediction == answer:
                    correct += 1
            else:
                results.append(None)
        return correct / total, correct, total, results

