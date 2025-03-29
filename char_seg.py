import cv2
import numpy as np
import os
from remove_bg import *
# Load the handwritten template image (keep original colors)
image_path = 'handwritten_sample_3.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if image is loaded properly
if image is None:
    raise FileNotFoundError(f"Error: Unable to read image file at {image_path}. Please check the file path.")

# Resize the image to 469x635
image = cv2.resize(image, (469, 635))

# Function to apply Unsharp Masking (sharpening)
def sharpen_image(img):
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    return sharpened

# Apply sharpening to the full image before segmentation
image = sharpen_image(image)

# Reference rectangle dimensions (from the manually drawn grid)
start_x, start_y = 13, 40
box_width, box_height = 69 - 13, 113 - 40  # Width and height of the reference box
difx, dify = box_width, box_height

# Character labels
char_labels = ['!', '"', "'", ',', '.', 'col', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
               'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
               'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# Generate all grid box coordinates
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
os.makedirs('segmented_letters/input', exist_ok=True)
os.makedirs('segmented_letters/output', exist_ok=True)


# Extract, crop, and save letters based on grid boxes
for i, (x1, y1, x2, y2) in enumerate(grid_boxes):
    letter = image[y1+12:y2-12, x1+6:x2-20]  # Crop from sharpened grayscale
    letter = cv2.bitwise_not(letter)
    letter = sharpen_image(letter)  # Apply sharpening before saving

    cv2.imwrite(f'segmented_letters/input/letter_{i}_{char_labels[i]}_.png', letter)
    # cv2.imwrite(f'segmented_letters/output/letter_{i}_{char_labels[i]}_.png', letter)

remover = BackgroundRemover(input_dir='segmented_letters/input', output_dir='segmented_letters/output')
remover.process_images()
print("✅ Handwriting letters segmented, sharpened, and saved!")
