import cv2
import numpy as np
import os
from PIL import Image
import imagehash
import string

# Paths
image_paths = [
    "extracted_images/image_1_1.jpeg",  # Uppercase
    "extracted_images/image_2_1.jpeg",  # Lowercase
]
base_save_path = "prepare_dataset/letters"
os.makedirs(base_save_path, exist_ok=True)

# Settings
MIN_HEIGHT = 35
new_height = 47
kernel = np.ones((17, 7), np.uint8)
start_point = [159, 30]
end_point = [1700, 129]
alphabet_sets = [string.ascii_uppercase, string.ascii_lowercase]
prefixes = ["upper_", "lower_"]

# Process both images
for img_idx, image_path in enumerate(image_paths):
    image = cv2.imread(image_path)
    alphabet = list(alphabet_sets[img_idx])
    prefix = prefixes[img_idx]

    for strip_idx in range(len(alphabet)):
        letter_label = alphabet[strip_idx]
        letter_dir = os.path.join(base_save_path, f"{prefix}{letter_label}")
        os.makedirs(letter_dir, exist_ok=True)

        # Crop the strip
        y1, y2 = start_point[1], end_point[1]
        cropped_strip = image[y1:y2, start_point[0]:end_point[0]]

        # Grayscale and threshold
        gray = cv2.cvtColor(cropped_strip, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda cnt: cv2.boundingRect(cnt)[0])

        existing_hashes = set()
        char_idx = 1

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h < MIN_HEIGHT:
                continue

            char_img = cropped_strip[y:y + h, x:x + w]
            pil_img = Image.fromarray(cv2.cvtColor(char_img, cv2.COLOR_BGR2RGB))
            hash_val = imagehash.phash(pil_img)
            if hash_val in existing_hashes:
                continue
            existing_hashes.add(hash_val)

            # Resize
            aspect_ratio = w / h
            new_width = int(new_height * aspect_ratio)
            resized = cv2.resize(char_img, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Save
            save_path = os.path.join(letter_dir, f"{char_idx}.png")
            cv2.imwrite(save_path, resized)
            print(f"Saved {prefix}{letter_label}/{char_idx}.png")
            char_idx += 1

        # Move rectangle down for next strip
        start_point[1] += 93
        end_point[1] += 93

    # Reset for lowercase image
    start_point = [159, 30]
    end_point = [1700, 129]
