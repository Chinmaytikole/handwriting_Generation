from rembg import remove
from PIL import Image
import io

# Load input image
input_path = 'generated_sentence.png'      # Replace with your image path
output_path = 'output_image.png'    # Path to save image without background

with open(input_path, 'rb') as inp_file:
    input_data = inp_file.read()

# Remove background
output_data = remove(input_data)

# Save output image
with open(output_path, 'wb') as out_file:
    out_file.write(output_data)

print("Background removed and saved to:", output_path)
