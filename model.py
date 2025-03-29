import os
import cv2
import numpy as np
from WordDetector.word_detector import prepare_img, detect, sort_line

# Load image
image = cv2.imread('resized_image.jpg')
image = cv2.resize(image, (1003, 1280))  # Resize image to 1003x1280

# Convert image to grayscale and ensure it's uint8
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image = np.uint8(gray_image)

height, width = gray_image.shape[:2]

# Define cropping parameters
start_point = [0, 42]
end_point = [1003, 100]
cropped_dir = "cropped_images"
detected_words_dir = "detected_words"
os.makedirs(cropped_dir, exist_ok=True)
os.makedirs(detected_words_dir, exist_ok=True)

# Process each cropped region
word_count = 0  # Counter for unique word filenames
for i in range(32):
    cropped_img = gray_image[start_point[1]+10:end_point[1]-10, start_point[0]:end_point[0]]
    if cropped_img.size == 0:
        continue

    # Resize cropped image to maintain aspect ratio with height 89
    aspect_ratio = cropped_img.shape[1] / cropped_img.shape[0]
    new_width = int(200 * aspect_ratio)
    resized_cropped_img = cv2.resize(cropped_img, (new_width, 200))

    # Save cropped image
    crop_filename = os.path.join(cropped_dir, f'cropped_{i}.jpg')
    cv2.imwrite(crop_filename, resized_cropped_img)

    # Prepare image for word detection
    prepared_img = prepare_img(resized_cropped_img, 50)

    # Detect words
    detections = detect(prepared_img, kernel_size=21, sigma=9, theta=5, min_area=100)
    sorted_lines = sort_line(detections)

    # Save detected words with fixed height of 30 pixels in a single directory
    if sorted_lines and sorted_lines[0]:
        for j, word in enumerate(sorted_lines[0]):
            word_aspect_ratio = word.img.shape[1] / word.img.shape[0]
            # word_new_width = int(30 * word_aspect_ratio)
            # resized_word = cv2.resize(word.img, (word_new_width, 30))  # Maintain aspect ratio with height 30
            word_filename = os.path.join(detected_words_dir, f'word_{word_count + 1}.jpg')
            cv2.imwrite(word_filename, word.img)
            word_count += 1

    # Update cropping positions
    start_point[1] += 43
    end_point[1] += 43
