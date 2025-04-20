from PIL import Image
import cv2
import numpy as np
import os
from handwriting_recognition import HandwrittenTextRecognizer
# Load the image
image_path = 'text_blank_page_2.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Create output directory
output_dir = 'test/words'
os.makedirs(output_dir, exist_ok=True)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply binary inverse thresholding
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Dilation to connect characters into words (horizontal structuring element)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))  # Adjust kernel size for spacing
dilated = cv2.dilate(thresh, kernel, iterations=1)

# Find external contours of the word blocks
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours by y (top to bottom), then x (left to right)
contours = sorted(contours, key=lambda cnt: (cv2.boundingRect(cnt)[1], cv2.boundingRect(cnt)[0]))

# Extract and save word images
word_images = []
for idx, cnt in enumerate(contours):
    x, y, w, h = cv2.boundingRect(cnt)

    # Optional: filter very small contours (noise)
    if w < 10 or h < 10:
        continue

    # Extract word using the bounding box
    word_img = image[y:y + h, x:x + w]
    pil_img = Image.fromarray(cv2.cvtColor(word_img, cv2.COLOR_BGR2RGB))
    recognizer = HandwrittenTextRecognizer()
    # Call the method with the PIL Image object to get the extracted text

    text = recognizer.recognize_text(pil_img)

    # Optional: Resize or normalize the word image
    resized_word_img = cv2.resize(word_img, (128, 32))  # Example size
    # Save the word image
    filename = os.path.join(output_dir, f"{text}.png")
    cv2.imwrite(filename, resized_word_img)
    word_images.append(resized_word_img)

    print(f"Extracted word {idx} and saved as {filename}")

# Display all extracted words (optional)

cv2.waitKey(0)
cv2.destroyAllWindows()
