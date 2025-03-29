from rembg import remove
from PIL import Image

# Input and output file paths
input_path = "handwriting_image_2.jpg"  # Replace with your image file
output_path = "output_image.png"  # Output will be in PNG format to support transparency

# Open the image and remove the background
with Image.open(input_path) as img:
    output = remove(img)
    output.save(output_path)

print(f"Background removed. Saved as {output_path}")
