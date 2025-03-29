import cv2
import numpy as np

# Load the image
image = cv2.imread('cropped_images/cropped_0.jpg', cv2.IMREAD_GRAYSCALE)

# Invert the image
inverted_image = cv2.bitwise_not(image)

# Define a horizontal kernel for dilation
kernel = np.ones((1, 45), np.uint8)  # Adjust the width (20) based on spacing

# Apply dilation
dilated_image = cv2.dilate(inverted_image, kernel, iterations=1)

# Save the output
cv2.imwrite('dilated_output.jpg', dilated_image)
