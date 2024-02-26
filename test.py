from PIL import Image
import numpy as np
from skimage import measure
from scipy import ndimage
from matplotlib import pyplot as plt

# Load the image from the path where your image is stored
img_path = 'cross_circles.png'  # Replace with your image file path
image = Image.open(img_path)

# Convert the image to grayscale
gray_image = image.convert('L')

# Convert the grayscale image to a numpy array
image_np = np.array(gray_image)

# Define a threshold to identify the circles and cross (plus sign)
# Assuming the background is white or light and the shapes are dark
threshold = 128

# Binarize the image: pixels with a value smaller than the threshold are black (True), the rest are white (False).
binary_image = image_np < threshold

# Label different objects in the image
labeled_image, num_features = ndimage.label(binary_image)
# print(labeled_image)
# display the labeled image in a plot
plt.imshow(labeled_image)
plt.show()
print(num_features)

# Measure properties of labeled regions
objects = measure.regionprops(labeled_image, intensity_image=image_np)

# The coordinates are given as (row, col), which corresponds to (y, x) in image coordinates
# We need to swap them to (x, y) to match typical Cartesian coordinates
positions = [(int(obj.centroid[1]), int(obj.centroid[0])) for obj in objects]

# Assuming that the plus sign will have a larger area than the circles
# Sort objects based on area and the last one will be the plus sign
sorted_objects = sorted(objects, key=lambda x: x.area, reverse=True)

# The largest area object is assumed to be the plus sign
plus_sign_position = (int(sorted_objects[0].centroid[1]), int(sorted_objects[0].centroid[0]))

# The rest are circles
circle_positions = [(int(obj.centroid[1]), int(obj.centroid[0])) for obj in sorted_objects[1:]]

print("Plus sign position:", plus_sign_position)
print("Circle positions:", circle_positions)