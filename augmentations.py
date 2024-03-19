import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance as dist
import math
from scipy import ndimage
from skimage import measure
import numpy as np

def RotateAugmentation(angle):
    """
    Rotate clockwise by the given angle.
    """
    def rotate(image):
        return ndimage.rotate(image, -angle, reshape=False)
    return rotate

def CropAugmentation(x, y, w, h):
    def crop(image):
        return image[y:y+h, x:x+w]
    return crop

def TranslateAugmentation(dx, dy):
    def translate(image):
        # return ndimage.shift(image, (dy, dx, 0))
        # Shift the image
        if len(image.shape) == 2:
            translated_image = ndimage.shift(image, (dy, dx), mode='nearest')
        else:
            translated_image = ndimage.shift(image, (dy, dx, 0), mode='nearest')

        blank_color = [255, 255, 255]
        blank_color = [0, 0, 0]
        blank_color = 0
        if dy > 0:  # Shifted down
            translated_image[:dy, :] = blank_color
        elif dy < 0:  # Shifted up
            translated_image[dy:, :] = blank_color

        if dx > 0:  # Shifted right
            translated_image[:, :dx] = blank_color
        elif dx < 0:  # Shifted left
            translated_image[:, dx:] = blank_color

        return translated_image
    return translate

def ScaleAugmentation(sx, sy):
    def scale(image):
        return cv2.resize(image, (0, 0), fx=sx, fy=sy)
    return scale

def ShearAugmentation(shear):
    def shear(image):
        shear_matrix = np.array([
            [1, shear],
            [0, 1]
        ])
        return cv2.warpAffine(image, shear_matrix, (image.shape[1], image.shape[0]))
    return shear

def ColorChangeAugmentation(shift):
    def color_change(image):
        return image + shift
    return color_change

def FlipAugmentation(angle):
    """
    Flip the image by the given angle, crossing the center of the image.
    Angle is in degrees. 0 is a vertical axis, going clockwise, 180 is a horizontal axis.
    """
    # Base angle type goes counterclockwise, reverse this
    angle = -angle
    def flip(image):
        # Calculate the center of the image
        center = np.array(image.shape[:2])[::-1] / 2.0
        
        # Compute the transformation matrix
        rotation_matrix = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
        
        # Perform the affine transformation
        rotated_image = cv2.warpAffine(image, rotation_matrix, image.shape[:2], flags=cv2.INTER_LINEAR)
        
        return rotated_image

    return flip

def augmentation_score(augmentation, org_img, dest_img):
    augmented_image = augmentation(org_img)
    # Rules that make an augmented imge good:
    # 1. It should be similar to the destination image
    # 2. It should have a similar structure to the original image (like number of colored pixels)
    l2 = np.mean((augmented_image - dest_img) ** 2)
    y_size = np.sum(dest_img > 0)
    augmented_size = np.sum(augmented_image > 0)
    if augmented_size == 0:
        return float('inf')
    return l2 * y_size / augmented_size
