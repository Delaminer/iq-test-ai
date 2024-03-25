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
        image = ndimage.rotate(image, -angle, reshape=False)
        # convert back to binary
        image = (image > 128)
        return image
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

def RotateTranslateAugmentation(angle, dx, dy):
    def rotate_translate(image):
        # Get image dimensions
        height, width = image.shape[:2]

        # Define rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), -angle, 1)

        # Apply rotation to image
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

        # Define translation matrix
        translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])

        # Apply translation to rotated image
        translated_rotated_image = cv2.warpAffine(rotated_image, translation_matrix, (width, height))

        return translated_rotated_image
    return rotate_translate

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

def ncc_score(image1, image2):
    image1_norm = image1.astype(float) / 255.0
    image2_norm = image2.astype(float) / 255.0
    mean1 = np.mean(image1_norm)
    mean2 = np.mean(image2_norm)
    image1_norm -= mean1
    image2_norm -= mean2
    cross_corr = np.sum(image1_norm * image2_norm)
    ncc_score = cross_corr / (np.sqrt(np.sum(image1_norm**2)) * np.sqrt(np.sum(image2_norm**2)))
    return ncc_score

def sift_score(image1, image2):
    if image1.dtype != np.uint8:
        image1 = image1.astype(np.uint8) * 255
    if image2.dtype != np.uint8:
        image2 = image2.astype(np.uint8) * 255
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    return len(good)

from skimage.metrics import structural_similarity as ssim
def warp_image(image1, image2):
    if image1.dtype != np.uint8:
        image1 = image1.astype(np.uint8) * 255
    if image2.dtype != np.uint8:
        image2 = image2.astype(np.uint8) * 255
    # Detect keypoints and compute descriptors
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    # Match keypoints
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Estimate transformation
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    # Apply transformation to img1
    aligned_img1 = cv2.warpPerspective(image1, M, (image2.shape[1], image2.shape[0]))

    # Compare similarity
    similarity = ssim(aligned_img1, image2)
    return similarity

def augmentation_score(augmentation, org_img, dest_img):
    augmented_image = augmentation(org_img)

    # Try returning just the number of pixels that the image has in common with the destination image

    # Rules that make an augmented imge good:
    # 1. It should be similar to the destination image
    # 2. It should have a similar structure to the original image (like number of colored pixels)
    # return sift_score(augmented_image, dest_img)
    loss = np.mean((augmented_image - dest_img) ** 2)
    penalty = np.sum(augmented_image > 0)
    # penalty = np.linalg.norm(augmented_image, ord=1)
    # penalty = 1
    # penalty = np.linalg.norm(augmented_image, ord=2)
    if penalty == 0:
        return float('inf')
    # return loss / penalty
    shared_pixels = np.sum(augmented_image == dest_img)
    shared_pixels = np.sum(np.bitwise_and(augmented_image, dest_img))
    if shared_pixels == 0:
        return float('inf')
    return 1 / shared_pixels
    # return 1 / ncc_score(augmented_image, dest_img)