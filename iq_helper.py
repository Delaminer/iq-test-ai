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
    grid = [cv2.bitwise_not(image[border:-border, border:-border]) for image in grid]

    if debug:
        print("Grid contours")
        show_image(draw_contours(image, grid_contours))

        print("Resulting grid images")
        show_images_grid(grid)

    choices = []
    return grid, choices

def force_to_binary(image):
    # if this is a color image, convert to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # if the image is not already binary, convert it to binary
    if len(np.unique(image)) > 2:
        _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    return image


# def get_contours(image, isGray = False, isBinary = False):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if not isGray else image
#     _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV) if not isBinary else (0, image)
#     contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     # it is possible a contour is found around the border of the image, so remove it
#     threshold = 0.95
#     contours = [contour for contour in contours if cv2.contourArea(contour) / (image.shape[0] * image.shape[1]) < threshold]
#     return contours