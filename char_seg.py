import cv2
import numpy as np
import os

# Load the handwritten template image
image = cv2.imread('Hanwritten_sample_5.jpg', cv2.IMREAD_GRAYSCALE)

# Apply thresholding to convert to binary
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

char_labels = ['!', '"', "'", ',', '.', 'col', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
               'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
               'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# Reference rectangle dimensions (from the manually drawn grid)
# (13, 40), (69, 111)
start_x, start_y = 13, 40
box_width, box_height = 69-13, 113-40  # Width and height of the reference box
difx, dify = box_width, box_height

# Generate all grid box coordinates used in drawing
grid_boxes = []
x1, y1 = start_x, start_y
for i in range(8):
    for j in range(8):
        if i < 2 and j >= 6:
            continue
        x2, y2 = x1 + box_width, y1 + box_height
        grid_boxes.append((x1, y1, x2, y2))
        x1 += difx
    y1 += dify
    x1 = start_x

# Create output directory
os.makedirs('segmented_letters/images', exist_ok=True)

# Extract, crop, and save letters based on grid boxes
for i, (x1, y1, x2, y2) in enumerate(grid_boxes):
    letter = binary_image[y1:y2, x1:x2]

    # Crop edges by 2 units
    # cropped_letter = letter[10:-10,20:-30]

    # Resize back to original dimensions
    # cropped_letter = cv2.resize(cropped_letter, (box_width, box_height))

    cv2.imwrite(f'segmented_letters/images/letter_{i}_{char_labels[i]}_.png', letter)

print("✅ Handwriting letters segmented, cropped, and saved!")
