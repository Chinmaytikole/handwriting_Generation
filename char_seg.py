import cv2
import numpy as np
import os

# Load the handwritten template image
image = cv2.imread('handwriting_sample_2.jpg', cv2.IMREAD_GRAYSCALE)

# Apply thresholding to convert to binary
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
char_labels = ['!', '"', "'", ',', '.', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
               'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
               'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
print(len(char_labels))
# Reference rectangle dimensions (from the manually drawn grid)
start_x, start_y = 20, 90
box_width, box_height = 127 - 20, 210 - 90  # Width and height of the reference box
difx, dify = box_width, box_height + 22


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

# Extract and save letters based on grid boxes
for i, (x1, y1, x2, y2) in enumerate(grid_boxes):

    letter = binary_image[y1:y2, x1:x2]
    letter = cv2.resize(letter, (box_width, box_height))
    cv2.imwrite(f'segmented_letters/images/letter_{i}_{char_labels[i]}_.png', letter)

print("✅ Handwriting letters segmented and saved!")
