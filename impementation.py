import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from torchvision import transforms


# Generator network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Downsampling layers
        self.down1 = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Upsampling layers
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 1, 4, 2, 1),
            nn.Tanh()
        )

        # Refine layer
        self.refine = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        # Apply bottleneck
        b = self.bottleneck(d4)

        u1 = self.up1(b)
        u2 = self.up2(torch.cat([u1, d3], 1))
        u3 = self.up3(torch.cat([u2, d2], 1))
        u4 = self.up4(torch.cat([u3, d1], 1))

        # Apply refine layer to the output
        refined = self.refine(u4)

        return refined


# Function to remove thin horizontal lines that often connect characters
def remove_connecting_lines(img_array, threshold=245, min_length=10, max_thickness=3):
    """
    Remove thin horizontal lines that might connect characters

    Args:
        img_array: NumPy array of grayscale image
        threshold: Pixel value below which is considered content (darker)
        min_length: Minimum length of line to consider for removal
        max_thickness: Maximum thickness of line to remove

    Returns:
        Cleaned image array
    """
    height, width = img_array.shape
    result = img_array.copy()

    # First create a binary image to identify potential lines
    binary = (img_array < threshold).astype(np.uint8)

    # Scan through rows of the image
    for y in range(height):
        # Check if this row is a potential thin line
        if y < height - max_thickness:
            # Check this row + rows below for a pattern that suggests a thin connecting line
            possible_line = True

            # Check if there's a dark line followed by lighter pixels below (typical connection pattern)
            for thickness in range(1, max_thickness + 1):
                if y + thickness < height:
                    row_dark_pixels = np.sum(binary[y])
                    below_dark_pixels = np.sum(binary[y + thickness])

                    # If the row has significantly more dark pixels than below, it might be a connecting line
                    # and has enough dark pixels to be considered a line (not just noise or part of a character)
                    if row_dark_pixels > min_length and row_dark_pixels > below_dark_pixels * 2:
                        # Now check if this is a thin line by examining the pixel pattern
                        run_lengths = []
                        in_run = False
                        run_length = 0

                        # Find continuous runs of dark pixels in this row
                        for x in range(width):
                            if binary[y, x] == 1:  # Dark pixel
                                if not in_run:
                                    in_run = True
                                    run_length = 1
                                else:
                                    run_length += 1
                            else:  # Light pixel
                                if in_run:
                                    in_run = False
                                    if run_length >= min_length:
                                        run_lengths.append(run_length)
                                    run_length = 0

                        # Don't forget the last run if it ends at the edge
                        if in_run and run_length >= min_length:
                            run_lengths.append(run_length)

                        # If we found any significant runs, consider this a connecting line
                        if run_lengths:
                            # Set this row to white (255) in the cleaned image
                            result[y, :] = 255
                            break

    return result


# Function to generate handwritten text using a pre-trained model and input letter images
def generate_handwritten_text(generator_path, input_letters_dir, text, output_path, spacing_factor=-0.3,
                              top_crop_percent=20, remove_lines=True, line_threshold=245):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize and load the generator
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()
    print(f"Generator model loaded from {generator_path}")

    # Ensure input letters directory exists
    if not os.path.exists(input_letters_dir):
        raise FileNotFoundError(f"Input letters directory not found: {input_letters_dir}")

    print(f"Using input letters from: {input_letters_dir}")

    # Define character order based on the provided information
    char_order = ['!', '"', "'", ',', '.', 'col', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                  'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
                  'g', 'h', 'i', 'j',
                  'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    # Special naming cases
    special_cases = {
        '?': 'letter_7_Quest_.png',
        ':': 'letter_5_col_.png'
    }

    # Load input letter images based on the naming convention
    letter_images = {}
    valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']

    # First, try to load all files with the specific format or special cases
    for i, char in enumerate(char_order):
        # Handle special cases
        if char == '?':
            filename = special_cases['?']
        elif char == 'col':
            filename = special_cases.get(':', f"letter_{i}_col_.png")
            char = ':'  # Map 'col' to ':' for the dictionary key
        else:
            filename = f"letter_{i}_{char}_.png"  # Regular format

        file_path = os.path.join(input_letters_dir, filename)

        if os.path.exists(file_path):
            try:
                img = Image.open(file_path).convert('L')  # Convert to grayscale
                letter_images[char] = img
                print(f"Loaded letter image for '{char}' from {filename}")
            except Exception as e:
                print(f"Error loading letter image '{file_path}': {e}")
        else:
            print(f"Warning: Image file not found for character '{char}' at path {file_path}")

            # Try alternate filenames for this character
            alternate_names = [
                f"letter_{i}_{char}.png",
                f"{char}.png"
            ]

            found = False
            for alt_name in alternate_names:
                alt_path = os.path.join(input_letters_dir, alt_name)
                if os.path.exists(alt_path):
                    try:
                        img = Image.open(alt_path).convert('L')
                        letter_images[char] = img
                        found = True
                        print(f"Loaded letter image for '{char}' from alternate file {alt_name}")
                        break
                    except Exception as e:
                        print(f"Error loading alternate image '{alt_path}': {e}")

            if not found:
                print(f"No valid image found for character '{char}'")

    # Fall back to scanning the directory for any remaining letters
    print("Scanning directory for additional letter images...")
    for filename in os.listdir(input_letters_dir):
        file_path = os.path.join(input_letters_dir, filename)

        # Skip directories and check if it's an image file
        if os.path.isdir(file_path) or not any(filename.lower().endswith(ext) for ext in valid_extensions):
            continue

        # Try to extract character from the filename based on various patterns
        if filename.startswith("letter_") and "_" in filename:
            parts = os.path.splitext(filename)[0].split('_')

            # Handle cases like letter_7_Quest_.png for ?
            if len(parts) >= 3 and "quest" in parts[2].lower():
                char = '?'
            # Handle cases like letter_5_col_.png for :
            elif len(parts) >= 3 and "col" in parts[2].lower():
                char = ':'
            # Handle standard cases
            elif len(parts) >= 3 and parts[2]:
                char = parts[2]
            else:
                continue

            if char not in letter_images:
                try:
                    img = Image.open(file_path).convert('L')  # Convert to grayscale
                    letter_images[char] = img
                    print(f"Loaded letter image for '{char}' from {filename}")
                except Exception as e:
                    print(f"Error loading letter image '{file_path}': {e}")

    # Check if we have at least some letters loaded
    if not letter_images:
        raise ValueError(f"No valid letter images found in {input_letters_dir}")

    # Report missing letters
    missing_chars = set()
    for char in text:
        if char != ' ' and char not in letter_images:
            missing_chars.add(char)

    if missing_chars:
        print(f"Warning: The following characters in your text have no input images: {', '.join(missing_chars)}")
        print("These will be represented by blank images.")

    # Transformation
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Create individual character images and transform them
    char_imgs = []
    for char in text:
        if char == ' ':
            # Handle spaces separately - make them narrower
            img_space = Image.new('L', (128, 128), 255)
            img_space_tensor = transform(img_space)
            char_imgs.append(img_space_tensor)
        else:
            # Use the loaded letter image if available, otherwise create a blank image
            if char in letter_images:
                img = letter_images[char]
            else:
                img = Image.new('L', (128, 128), 255)

            img_tensor = transform(img)
            char_imgs.append(img_tensor)

    # Generate handwritten versions
    handwritten_chars = []
    with torch.no_grad():
        for img_tensor in char_imgs:
            input_tensor = img_tensor.unsqueeze(0).to(device)
            fake = generator(input_tensor)
            fake_img = (fake.cpu().squeeze().numpy() * 0.5 + 0.5) * 255

            # Apply line removal if requested
            if remove_lines:
                fake_img = remove_connecting_lines(fake_img, threshold=line_threshold)

            handwritten_chars.append(fake_img)

    # Character boundary detection
    # Use a lower threshold to detect even faint content
    threshold = 245
    char_info = []

    for i, img_array in enumerate(handwritten_chars):
        if text[i] == ' ':
            # Handle spaces - make them very narrow or even negative for overlap
            char_info.append({
                'char': ' ',
                'cropped': None,
                'width': -70,  # Negative width will cause words to overlap slightly
                'height': 0,
                'y_offset': 0,
                'left_edge': 0,
                'right_edge': 0
            })
            continue

        # Find the content boundaries with a lower threshold for more aggressive cropping
        binary = (img_array < threshold).astype(np.uint8)

        # Find content boundaries
        rows_with_content = np.any(binary, axis=1)
        cols_with_content = np.any(binary, axis=0)

        if np.any(rows_with_content) and np.any(cols_with_content):
            # Find the first row with content
            top_edge = np.argmax(rows_with_content)
            bottom_edge = img_array.shape[0] - np.argmax(rows_with_content[::-1])
            left_edge = np.argmax(cols_with_content)
            right_edge = img_array.shape[1] - np.argmax(cols_with_content[::-1])

            # Apply additional top cropping by moving the top edge down
            # Calculate how many pixels to crop from the top (based on percentage)
            additional_top_crop = int((bottom_edge - top_edge) * top_crop_percent / 100)
            top_edge += additional_top_crop

            # Make sure we still have a valid region after cropping
            if top_edge >= bottom_edge:
                top_edge = bottom_edge - 1

            # Create a copy of the image region after the top has been cropped
            cropped = img_array[top_edge:bottom_edge, left_edge:right_edge].copy()

            # Convert to PIL Image with alpha channel (RGBA)
            # Create an alpha mask where white becomes transparent
            alpha = np.where(cropped > threshold, 0, 255).astype(np.uint8)
            # Create an RGBA image where the RGB channels are black (0) and A is our alpha mask
            rgba = np.zeros((cropped.shape[0], cropped.shape[1], 4), dtype=np.uint8)
            # Set the RGB values to the inverted grayscale values (so darker pixels remain visible)
            for c in range(3):  # RGB channels
                rgba[:, :, c] = 255 - cropped
            # Set the alpha channel
            rgba[:, :, 3] = alpha

            # Convert to PIL Image
            cropped_img = Image.fromarray(rgba)

            char_info.append({
                'char': text[i],
                'cropped': cropped_img,
                'width': cropped.shape[1],
                'height': cropped.shape[0],
                'y_offset': top_edge,
                'left_edge': left_edge,
                'right_edge': right_edge
            })
        else:
            # Empty character
            char_info.append({
                'char': text[i],
                'cropped': None,
                'width': 5,
                'height': 0,
                'y_offset': 0,
                'left_edge': 0,
                'right_edge': 0
            })

    # Calculate kerning adjustments - how much to overlap characters
    for i in range(1, len(char_info)):
        # Skip if either character is a space or empty
        if char_info[i]['cropped'] is None or char_info[i - 1]['cropped'] is None:
            continue

        # Determine overlap amount based on the character widths and spacing factor
        # Negative spacing_factor means characters overlap
        char_info[i]['overlap'] = int(char_info[i]['width'] * abs(spacing_factor)) if spacing_factor < 0 else 0

    # Calculate total width considering overlaps
    total_width = 0
    for i, info in enumerate(char_info):
        if i > 0 and 'overlap' in info:
            total_width += info['width'] - info['overlap']
        else:
            total_width += info['width']

    total_width += 10  # Add small margin
    max_height = max(info['height'] for info in char_info if info['height'] > 0)

    # Create a new image with alpha channel
    result_img = Image.new('RGBA', (total_width, max_height + 10), (0, 0, 0, 0))  # Transparent background

    # Paste characters with overlapping
    x_offset = 5  # Left margin
    baseline = max_height * 0.8  # Approximate baseline for text

    for i, info in enumerate(char_info):
        if info['cropped'] is not None:
            # Apply kerning/overlap adjustment for characters after the first one
            if i > 0 and 'overlap' in info:
                x_offset -= info['overlap']

            # Calculate vertical position with baseline alignment
            y_offset = int(baseline - info['height'] * 0.8)

            # Paste the character
            result_img.paste(info['cropped'], (x_offset, y_offset), info['cropped'])

        # Update x_offset
        x_offset += info['width']

    # Save the result
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    result_img.save(output_path)
    print(f"Generated handwritten text saved to {output_path}")

    return result_img


if __name__ == "__main__":
    # Your specific parameters
    generator_path = "handwriting_gan_output/generator_final.pth"
    input_letters_dir = "segmented_letters/output"  # Directory containing letter images
    text = "Hello hoW Are You?"
    output_path = "improved_hello_world.png"
    spacing = 0  # Character spacing within words
    top_crop = 20
    remove_lines = True
    line_threshold = 240

    # Generate handwritten text with input letter images
    try:
        result = generate_handwritten_text(
            generator_path=generator_path,
            input_letters_dir=input_letters_dir,
            text=text,
            output_path=output_path,
            spacing_factor=spacing,
            top_crop_percent=top_crop,
            remove_lines=remove_lines,
            line_threshold=line_threshold
        )
        print(f"Handwritten text saved to: {output_path}")
    except Exception as e:
        print(f"Error generating handwritten text: {e}")