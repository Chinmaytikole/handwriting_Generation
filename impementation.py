# import os
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset
# from PIL import Image, ImageDraw, ImageFont, ImageOps
# import numpy as np
# from torchvision import transforms
#
#
# # Define the Generator network (needs to match exactly what was used during training)
# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         # Downsampling layers
#         self.down1 = nn.Sequential(
#             nn.Conv2d(1, 64, 4, 2, 1),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
#         self.down2 = nn.Sequential(
#             nn.Conv2d(64, 128, 4, 2, 1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
#         self.down3 = nn.Sequential(
#             nn.Conv2d(128, 256, 4, 2, 1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
#         self.down4 = nn.Sequential(
#             nn.Conv2d(256, 512, 4, 2, 1),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
#
#         # Upsampling layers
#         self.up1 = nn.Sequential(
#             nn.ConvTranspose2d(512, 256, 4, 2, 1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True)
#         )
#         self.up2 = nn.Sequential(
#             nn.ConvTranspose2d(512, 128, 4, 2, 1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True)
#         )
#         self.up3 = nn.Sequential(
#             nn.ConvTranspose2d(256, 64, 4, 2, 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True)
#         )
#         self.up4 = nn.Sequential(
#             nn.ConvTranspose2d(128, 1, 4, 2, 1),
#             nn.Tanh()
#         )
#
#     def forward(self, x):
#         d1 = self.down1(x)
#         d2 = self.down2(d1)
#         d3 = self.down3(d2)
#         d4 = self.down4(d3)
#
#         u1 = self.up1(d4)
#         u2 = self.up2(torch.cat([u1, d3], 1))
#         u3 = self.up3(torch.cat([u2, d2], 1))
#         u4 = self.up4(torch.cat([u3, d1], 1))
#
#         return u4
#
#
# # Function to remove thin horizontal lines that often connect characters
# def remove_connecting_lines(img_array, threshold=245, min_length=10, max_thickness=3):
#     """
#     Remove thin horizontal lines that might connect characters
#
#     Args:
#         img_array: NumPy array of grayscale image
#         threshold: Pixel value below which is considered content (darker)
#         min_length: Minimum length of line to consider for removal
#         max_thickness: Maximum thickness of line to remove
#
#     Returns:
#         Cleaned image array
#     """
#     height, width = img_array.shape
#     result = img_array.copy()
#
#     # First create a binary image to identify potential lines
#     binary = (img_array < threshold).astype(np.uint8)
#
#     # Scan through rows of the image
#     for y in range(height):
#         # Check if this row is a potential thin line
#         if y < height - max_thickness:
#             # Check this row + rows below for a pattern that suggests a thin connecting line
#             possible_line = True
#
#             # Check if there's a dark line followed by lighter pixels below (typical connection pattern)
#             for thickness in range(1, max_thickness + 1):
#                 if y + thickness < height:
#                     row_dark_pixels = np.sum(binary[y])
#                     below_dark_pixels = np.sum(binary[y + thickness])
#
#                     # If the row has significantly more dark pixels than below, it might be a connecting line
#                     # and has enough dark pixels to be considered a line (not just noise or part of a character)
#                     if row_dark_pixels > min_length and row_dark_pixels > below_dark_pixels * 2:
#                         # Now check if this is a thin line by examining the pixel pattern
#                         run_lengths = []
#                         in_run = False
#                         run_length = 0
#
#                         # Find continuous runs of dark pixels in this row
#                         for x in range(width):
#                             if binary[y, x] == 1:  # Dark pixel
#                                 if not in_run:
#                                     in_run = True
#                                     run_length = 1
#                                 else:
#                                     run_length += 1
#                             else:  # Light pixel
#                                 if in_run:
#                                     in_run = False
#                                     if run_length >= min_length:
#                                         run_lengths.append(run_length)
#                                     run_length = 0
#
#                         # Don't forget the last run if it ends at the edge
#                         if in_run and run_length >= min_length:
#                             run_lengths.append(run_length)
#
#                         # If we found any significant runs, consider this a connecting line
#                         if run_lengths:
#                             # Set this row to white (255) in the cleaned image
#                             result[y, :] = 255
#                             break
#
#     return result
#
#
# # Function to generate handwritten text on lined paper template
# def generate_text_on_lined_paper(generator_path, ttf_path, text, paper_template_path, output_path,
#                                  spacing_factor=-0.3, top_crop_percent=20, remove_lines=True,
#                                  line_threshold=245, max_chars_per_line=40, size_multiplier=1.5,
#                                  invert_colors=True):  # New parameter to control color inversion
#     # Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#
#     # Initialize and load the generator
#     generator = Generator().to(device)
#     generator.load_state_dict(torch.load(generator_path, map_location=device))
#     generator.eval()
#     print(f"Generator model loaded from {generator_path}")
#
#     # Load font
#     try:
#         # Increase base font size for larger characters
#         font = ImageFont.truetype(ttf_path, int(64 * size_multiplier))
#         print(f"Font loaded from {ttf_path} with increased size")
#     except Exception as e:
#         print(f"Error loading font: {e}")
#         print("Using default font instead")
#         font = ImageFont.load_default()
#
#     # Load paper template
#     try:
#         paper_template = Image.open(paper_template_path).convert('RGBA')
#         print(f"Paper template loaded from {paper_template_path}")
#     except Exception as e:
#         print(f"Error loading paper template: {e}")
#         print("Creating blank template instead")
#         paper_template = Image.new('RGBA', (800, 1000), (255, 255, 255, 255))
#
#     # Analyze paper template to find lines
#     template_gray = np.array(paper_template.convert('L'))
#     template_width, template_height = paper_template.size
#
#     # Detect horizontal lines in the template
#     line_positions = []
#     threshold = 200  # Threshold for detecting lines
#
#     # Compute the average brightness of each row
#     row_brightness = np.mean(template_gray, axis=1)
#
#     # Find rows that are significantly darker (likely to be lines)
#     for y in range(1, template_height - 1):
#         if (row_brightness[y] < threshold and
#                 row_brightness[y] < row_brightness[y - 1] and
#                 row_brightness[y] < row_brightness[y + 1]):
#             line_positions.append(y)
#
#     # If no lines were detected, create evenly spaced lines
#     if not line_positions:
#         print("No lines detected in template, creating evenly spaced lines")
#         line_spacing = 30  # Default line spacing
#         line_positions = list(range(60, template_height - 60, line_spacing))
#
#     # Calculate the average line spacing
#     line_spacing = np.mean(np.diff(line_positions)) if len(line_positions) > 1 else 30
#     print(f"Detected {len(line_positions)} lines with average spacing of {line_spacing:.2f} pixels")
#
#     # Adjust max_chars_per_line based on size_multiplier
#     max_chars_per_line = int(max_chars_per_line / size_multiplier)
#
#     # Split text into lines that fit on the paper
#     words = text.split()
#     lines = []
#     current_line = []
#     current_length = 0
#
#     for word in words:
#         if current_length + len(word) + len(current_line) <= max_chars_per_line:
#             current_line.append(word)
#             current_length += len(word)
#         else:
#             lines.append(' '.join(current_line))
#             current_line = [word]
#             current_length = len(word)
#
#     if current_line:
#         lines.append(' '.join(current_line))
#
#     # Transformation for the generator
#     transform = transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5])
#     ])
#
#     # Create a copy of the paper template for the result
#     result_img = paper_template.copy()
#
#     # Process each line of text
#     for line_idx, line_text in enumerate(lines):
#         if line_idx >= len(line_positions) - 1:
#             print(f"Warning: More lines of text than available lines on template. Skipping remaining text.")
#             break
#
#         # Skip every other line if characters are too large for consecutive lines
#         target_line_idx = line_idx * 2 if size_multiplier > 1.3 else line_idx
#
#         if target_line_idx >= len(line_positions) - 1:
#             print(f"Warning: Ran out of space on template. Skipping remaining text.")
#             break
#
#         # Get the target y-position for this line (align with the ruled line)
#         target_y = line_positions[target_line_idx]
#
#         # Create individual character images and transform them
#         char_imgs = []
#         for char in line_text:
#             if char == ' ':
#                 # Handle spaces separately - make them narrower
#                 img_space = Image.new('L', (128, 128), 255)
#                 img_space_tensor = transform(img_space)
#                 char_imgs.append(img_space_tensor)
#             else:
#                 # Create a larger canvas for bigger characters
#                 canvas_size = int(128 * size_multiplier)
#                 img = Image.new('L', (canvas_size, canvas_size), 255)
#                 draw = ImageDraw.Draw(img)
#
#                 # Get text size
#                 try:
#                     left, top, right, bottom = font.getbbox(char)
#                     w, h = right - left, bottom - top
#                 except AttributeError:
#                     # Fallback for older PIL versions
#                     try:
#                         w, h = font.getsize(char)
#                     except:
#                         w, h = int(30 * size_multiplier), int(40 * size_multiplier)  # Default size
#
#                 draw.text(((canvas_size - w) // 2, (canvas_size - h) // 2), char, font=font, fill=0)
#
#                 # Resize back to 128x128 for the generator
#                 img = img.resize((128, 128), Image.LANCZOS)
#                 img_tensor = transform(img)
#                 char_imgs.append(img_tensor)
#
#         # Generate handwritten versions
#         handwritten_chars = []
#         with torch.no_grad():
#             for img_tensor in char_imgs:
#                 input_tensor = img_tensor.unsqueeze(0).to(device)
#                 fake = generator(input_tensor)
#                 fake_img = (fake.cpu().squeeze().numpy() * 0.5 + 0.5) * 255
#
#                 # Apply line removal if requested
#                 if remove_lines:
#                     fake_img = remove_connecting_lines(fake_img, threshold=line_threshold)
#
#                 handwritten_chars.append(fake_img)
#
#         # Character boundary detection
#         threshold = 245
#         char_info = []
#
#         for i, img_array in enumerate(handwritten_chars):
#             if line_text[i] == ' ':
#                 # Handle spaces - adjust width based on size_multiplier
#                 char_info.append({
#                     'char': ' ',
#                     'cropped': None,
#                     'width': int(20 * size_multiplier),  # Wider spaces for larger text
#                     'height': 0,
#                     'y_offset': 0,
#                     'left_edge': 0,
#                     'right_edge': 0
#                 })
#                 continue
#
#             # Find the content boundaries
#             binary = (img_array < threshold).astype(np.uint8)
#
#             # Find content boundaries
#             rows_with_content = np.any(binary, axis=1)
#             cols_with_content = np.any(binary, axis=0)
#
#             if np.any(rows_with_content) and np.any(cols_with_content):
#                 # Find the first row with content
#                 top_edge = np.argmax(rows_with_content)
#                 bottom_edge = img_array.shape[0] - np.argmax(rows_with_content[::-1])
#                 left_edge = np.argmax(cols_with_content)
#                 right_edge = img_array.shape[1] - np.argmax(cols_with_content[::-1])
#
#                 # Apply additional top cropping
#                 additional_top_crop = int((bottom_edge - top_edge) * top_crop_percent / 100)
#                 top_edge += additional_top_crop
#
#                 # Make sure we still have a valid region after cropping
#                 if top_edge >= bottom_edge:
#                     top_edge = bottom_edge - 1
#
#                 # Create a copy of the image region
#                 cropped = img_array[top_edge:bottom_edge, left_edge:right_edge].copy()
#
#                 # Apply color inversion if requested
#                 if invert_colors:
#                     cropped = 255 - cropped
#
#                 # Convert to PIL Image with alpha channel (RGBA)
#                 if invert_colors:
#                     # For inverted colors, lighter pixels are more transparent
#                     alpha = np.where(cropped < threshold, 0, 255).astype(np.uint8)
#                     rgba = np.zeros((cropped.shape[0], cropped.shape[1], 4), dtype=np.uint8)
#                     for c in range(3):  # RGB channels
#                         rgba[:, :, c] = cropped  # Use inverted values directly
#                 else:
#                     # Original behavior for non-inverted colors
#                     alpha = np.where(cropped > threshold, 0, 255).astype(np.uint8)
#                     rgba = np.zeros((cropped.shape[0], cropped.shape[1], 4), dtype=np.uint8)
#                     for c in range(3):  # RGB channels
#                         rgba[:, :, c] = 255 - cropped
#
#                 rgba[:, :, 3] = alpha
#                 cropped_img = Image.fromarray(rgba)

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
from torchvision import transforms
import random
import math


# Define the Generator network (same architecture as training)
class Generator(nn.Module):
    def _init_(self):
        super(Generator, self)._init_()
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

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u1 = self.up1(d4)
        u2 = self.up2(torch.cat([u1, d3], 1))
        u3 = self.up3(torch.cat([u2, d2], 1))
        u4 = self.up4(torch.cat([u3, d1], 1))

        return u4


# Enhanced function to remove connecting lines with more natural preservation of character features
def smart_line_removal(img_array, threshold=240, min_length=10, max_thickness=3):
    """
    Enhanced function to remove horizontal connecting lines while preserving natural character features

    Args:
        img_array: NumPy array of grayscale image
        threshold: Pixel value threshold for line detection
        min_length: Minimum length of line to consider for removal
        max_thickness: Maximum thickness of line to remove

    Returns:
        Cleaned image array with natural connections preserved
    """
    height, width = img_array.shape
    result = img_array.copy()
    binary = (img_array < threshold).astype(np.uint8)

    # First pass: Identify potential line segments
    potential_lines = []

    for y in range(height):
        if y < height - max_thickness:
            # Analyze row features
            row_segments = []
            run_start = -1

            for x in range(width):
                if binary[y, x] == 1:  # Dark pixel
                    if run_start == -1:
                        run_start = x
                elif run_start != -1:  # End of dark run
                    run_length = x - run_start
                    if run_length >= min_length:
                        # Check if this might be a connecting line
                        is_connecting_line = True

                        # Check surrounding context (above and below)
                        for dy in [-2, -1, 1, 2]:
                            if 0 <= y + dy < height:
                                # Count dark pixels in the same horizontal span
                                context_pixels = np.sum(binary[y + dy, run_start:x])
                                # If many dark pixels above/below, it's likely part of a character, not a line
                                if context_pixels > run_length * 0.4:
                                    is_connecting_line = False
                                    break

                        if is_connecting_line:
                            row_segments.append((run_start, x))

                    run_start = -1

            # Don't forget the last run if it ends at the edge
            if run_start != -1:
                run_length = width - run_start
                if run_length >= min_length:
                    row_segments.append((run_start, width))

            if row_segments:
                potential_lines.append((y, row_segments))

    # Second pass: Remove confirmed lines with a natural fade at edges
    for y, segments in potential_lines:
        for start, end in segments:
            # Create a natural fade at the edges of the removal (5px fade)
            fade_length = min(5, (end - start) // 4)

            # Apply full removal to center
            result[y, start + fade_length:end - fade_length] = 255

            # Apply graduated fade at edges
            for i in range(fade_length):
                # Calculate fade factor (0.0 to 1.0)
                fade_factor = i / fade_length

                # Apply faded removal to left edge
                left_pos = start + i
                if 0 <= left_pos < width:
                    original = result[y, left_pos]
                    result[y, left_pos] = original + int((255 - original) * fade_factor)

                # Apply faded removal to right edge
                right_pos = end - i - 1
                if 0 <= right_pos < width:
                    original = result[y, right_pos]
                    result[y, right_pos] = original + int((255 - original) * fade_factor)

    return result


# New function to add natural handwriting variations
def apply_natural_variations(img_array, variation_level=0.5):
    """
    Apply natural variations to make the handwriting look more human

    Args:
        img_array: NumPy array of grayscale image
        variation_level: Level of variation to apply (0.0 to 1.0)

    Returns:
        Modified image array with natural variations
    """
    height, width = img_array.shape
    result = img_array.copy()

    # Convert to PIL for filter operations
    img_pil = Image.fromarray(result.astype(np.uint8))

    # 1. Apply slight Gaussian blur to simulate ink bleeding/spreading
    blur_radius = 0.3 + (0.4 * variation_level)
    img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # 2. Apply slight noise to simulate paper texture
    noise_array = np.zeros((height, width), dtype=np.uint8)
    noise_level = int(5 * variation_level)
    if noise_level > 0:
        noise_array = np.random.randint(0, noise_level, (height, width), dtype=np.uint8)

    # Convert back to numpy and add noise
    result = np.array(img_pil)

    # Only add noise to areas near the text (darker regions)
    text_mask = (result < 220).astype(np.float32)

    # Dilate the mask slightly to get the areas around text
    from scipy.ndimage import binary_dilation
    influence_mask = binary_dilation(text_mask, iterations=3).astype(np.float32)

    # Apply noise with distance-based influence
    result = result + (noise_array * influence_mask)

    # Ensure valid range
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


# New function to apply natural pressure variations
def apply_pressure_variations(img_pil, pressure_variation=0.4):
    """
    Apply natural pressure variations to the handwriting

    Args:
        img_pil: PIL Image of the character
        pressure_variation: Amount of pressure variation (0.0 to 1.0)

    Returns:
        Modified PIL Image with pressure variations
    """
    if pressure_variation <= 0:
        return img_pil

    # Convert to numpy for processing
    img_array = np.array(img_pil)

    # If this is an RGBA image, process alpha channel
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        # Split channels
        rgb = img_array[:, :, :3]
        alpha = img_array[:, :, 3].copy()

        # Create a pressure map (darker in some areas, lighter in others)
        h, w = alpha.shape

        # Create a smooth gradient for natural pressure variation
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x, y)

        # Generate several random points to create natural pressure centers
        num_centers = 2 + int(pressure_variation * 3)
        pressure_map = np.zeros((h, w), dtype=np.float32)

        for _ in range(num_centers):
            # Random center position
            cx = random.uniform(-1, 1)
            cy = random.uniform(-1, 1)
            # Random intensity and spread
            intensity = random.uniform(0.5, 1.0) * pressure_variation * 50
            spread = random.uniform(0.3, 0.8)

            # Create a gaussian peak at this center
            peak = intensity * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / spread)
            pressure_map += peak

        # Normalize pressure map
        pressure_map = np.clip(pressure_map, 0, 255)

        # Apply pressure map to alpha channel where text exists
        text_mask = (alpha > 0)
        alpha[text_mask] = np.clip(alpha[text_mask] - pressure_map[text_mask], 0, 255)

        # Reconstruct image
        result = np.dstack((rgb, alpha))
        return Image.fromarray(result)

    return img_pil


# New function to simulate natural pen movement
def apply_natural_pen_movement(char_info_list, movement_variation=0.4):
    """
    Apply natural pen movement variations to a sequence of characters

    Args:
        char_info_list: List of character info dictionaries
        movement_variation: Amount of movement variation (0.0 to 1.0)

    Returns:
        Modified char_info_list with updated positions
    """
    if not char_info_list or movement_variation <= 0:
        return char_info_list

    # Natural writing has subtle vertical variations (baseline shifts)
    baseline_shift = 0

    for i in range(len(char_info_list)):
        # Skip spaces
        if char_info_list[i]['char'] == ' ' or char_info_list[i]['cropped'] is None:
            # Reset baseline shift subtly after spaces
            baseline_shift = baseline_shift * 0.8
            continue

        # Apply baseline shift
        shift_strength = movement_variation * 4  # pixels

        # Random walk for baseline
        baseline_shift += random.uniform(-shift_strength / 2, shift_strength / 2)
        # Limit maximum shift
        baseline_shift = np.clip(baseline_shift, -shift_strength * 2, shift_strength * 2)

        # Apply the shift (will be used during pasting)
        char_info_list[i]['baseline_shift'] = int(baseline_shift)

        # Also vary character rotation slightly
        if random.random() < 0.7:  # 70% chance to rotate
            rotation_angle = random.uniform(-2 * movement_variation, 2 * movement_variation)
            char_info_list[i]['rotation'] = rotation_angle
        else:
            char_info_list[i]['rotation'] = 0

    return char_info_list


# New function to apply natural slant variation
def apply_slant_variation(img_pil, slant_angle=0):
    """
    Apply a natural slant to a character

    Args:
        img_pil: PIL Image of the character
        slant_angle: Angle to slant in degrees

    Returns:
        Slanted PIL Image
    """
    if slant_angle == 0:
        return img_pil

    # Convert angle to radians
    angle_rad = math.radians(slant_angle)

    # Calculate shear factor
    shear_factor = math.tan(angle_rad)

    # Apply affine transform for slant
    width, height = img_pil.size

    # Create transform matrix for shear
    transform_matrix = (1, shear_factor, -shear_factor * height / 2, 0, 1, 0)

    # Apply transform
    slanted_img = img_pil.transform(
        (width, height),
        Image.AFFINE,
        transform_matrix,
        resample=Image.BICUBIC
    )

    return slanted_img


# Function to estimate natural stroke width for a character
def estimate_stroke_width(img_array, threshold=200):
    """
    Estimate the average stroke width in a character image

    Args:
        img_array: NumPy array of grayscale image
        threshold: Pixel value threshold for considering part of stroke

    Returns:
        Estimated average stroke width
    """
    binary = (img_array < threshold).astype(np.uint8)

    # If no content, return default
    if not np.any(binary):
        return 2

    # Find horizontal and vertical runs of dark pixels
    h_runs = []
    v_runs = []

    # Check horizontal runs
    height, width = binary.shape
    for y in range(height):
        run_start = -1
        for x in range(width):
            if binary[y, x] == 1:
                if run_start == -1:
                    run_start = x
            elif run_start != -1:
                run_length = x - run_start
                if 1 < run_length < 20:  # Ignore very thin or thick runs
                    h_runs.append(run_length)
                run_start = -1
        # Check for run at edge
        if run_start != -1:
            run_length = width - run_start
            if 1 < run_length < 20:
                h_runs.append(run_length)

    # Check vertical runs
    for x in range(width):
        run_start = -1
        for y in range(height):
            if binary[y, x] == 1:
                if run_start == -1:
                    run_start = y
            elif run_start != -1:
                run_length = y - run_start
                if 1 < run_length < 20:
                    v_runs.append(run_length)
                run_start = -1
        # Check for run at edge
        if run_start != -1:
            run_length = height - run_start
            if 1 < run_length < 20:
                v_runs.append(run_length)

    # Calculate average stroke width
    if h_runs or v_runs:
        avg_h = np.mean(h_runs) if h_runs else 0
        avg_v = np.mean(v_runs) if v_runs else 0

        # Horizontal runs are usually longer than vertical in Latin script
        if avg_h > 0 and avg_v > 0:
            return min(avg_h, avg_v)
        return max(avg_h, avg_v)

    return 2  # Default if no runs found


# New function for consistency checking between adjacent characters
def adjust_for_consistency(char_info_list):
    """
    Adjust character properties for better consistency across a word

    Args:
        char_info_list: List of character info dictionaries

    Returns:
        Modified char_info_list with more consistent properties
    """
    if not char_info_list:
        return char_info_list

    # Calculate stroke widths for all characters
    stroke_widths = []
    for info in char_info_list:
        if info['char'] != ' ' and info['cropped'] is not None:
            img_array = np.array(info['cropped'].convert('L'))
            stroke_widths.append(estimate_stroke_width(img_array))

    # If no valid stroke widths, return unchanged
    if not stroke_widths:
        return char_info_list

    # Calculate median stroke width for consistency
    median_stroke = np.median(stroke_widths)

    # Apply consistency adjustments
    for i in range(len(char_info_list)):
        if char_info_list[i]['char'] == ' ' or char_info_list[i]['cropped'] is None:
            continue

        # Adjust overlap for consistency based on characters
        if i > 0 and 'overlap' in char_info_list[i]:
            # Certain character pairs should have more/less overlap
            prev_char = char_info_list[i - 1]['char']
            curr_char = char_info_list[i]['char']

            # Character pairs that typically have more overlap
            more_overlap_pairs = [
                ('r', 'a'), ('r', 'e'), ('r', 'o'), ('r', 'u'),
                ('v', 'a'), ('v', 'e'), ('v', 'o'), ('v', 'u'),
                ('w', 'a'), ('w', 'e'), ('w', 'o'), ('w', 'u'),
                ('T', 'a'), ('T', 'e'), ('T', 'o'), ('T', 'u')
            ]

            # Character pairs that typically have less overlap
            less_overlap_pairs = [
                ('o', 'a'), ('o', 'e'), ('o', 'o'), ('o', 'u'),
                ('b', 'a'), ('b', 'e'), ('b', 'o'), ('b', 'u'),
                ('d', 'a'), ('d', 'e'), ('d', 'o'), ('d', 'u')
            ]

            # Adjust overlap based on character pairs
            if (prev_char, curr_char) in more_overlap_pairs:
                char_info_list[i]['overlap'] = int(char_info_list[i]['overlap'] * 1.3)
            elif (prev_char, curr_char) in less_overlap_pairs:
                char_info_list[i]['overlap'] = int(char_info_list[i]['overlap'] * 0.7)

    return char_info_list


# Function to create natural handwriting rhythm patterns
def create_handwriting_rhythm(text_length, rhythm_strength=0.5):
    """
    Create natural rhythm patterns for handwriting

    Args:
        text_length: Length of text to create rhythm for
        rhythm_strength: Strength of the rhythm pattern (0.0 to 1.0)

    Returns:
        List of rhythm factors for each character position
    """
    if rhythm_strength <= 0:
        return [1.0] * text_length

    # People naturally write in rhythmic patterns
    # This simulates the slight variations in pressure, spacing, and size

    # Create a base sine wave pattern
    base_frequency = random.uniform(0.1, 0.2)  # Characters per cycle
    rhythm = []

    for i in range(text_length):
        # Base rhythm
        base = math.sin(i * base_frequency * 2 * math.pi)

        # Add some noise/variation
        noise = random.uniform(-0.3, 0.3)

        # Combine with controlled strength
        factor = 1.0 + (base + noise) * rhythm_strength * 0.15
        rhythm.append(factor)

    return rhythm


# Improved function to generate handwritten text on lined paper
def generate_text_on_lined_paper(generator_path, ttf_path, text, paper_template_path, output_path,
                                 spacing_factor=-0.3, top_crop_percent=20, remove_lines=True,
                                 line_threshold=245, max_chars_per_line=40,
                                 handwriting_style=None):
    """
    Generate handwritten text on lined paper with enhanced natural features

    Args:
        generator_path: Path to the trained GAN generator model
        ttf_path: Path to the font file for initial character generation
        text: Text to render as handwriting
        paper_template_path: Path to the lined paper template image
        output_path: Where to save the final image
        spacing_factor: Character spacing within words (negative for overlap)
        top_crop_percent: Percentage of top to crop (for alignment)
        remove_lines: Whether to remove artificial connecting lines
        line_threshold: Threshold for line removal
        max_chars_per_line: Maximum characters per line of paper
        handwriting_style: Dictionary of style parameters (optional)
    """
    # Set default style if none provided
    if handwriting_style is None:
        handwriting_style = {
            'line_removal_strength': 0.8,  # How aggressively to remove connecting lines (0-1)
            'pressure_variation': 0.4,  # Variation in pen pressure (0-1)
            'movement_variation': 0.5,  # Variation in pen movement/baseline (0-1)
            'slant_angle': random.uniform(-2, 5),  # Overall slant of the handwriting
            'rhythm_strength': 0.5,  # Strength of natural writing rhythm (0-1)
            'variation_level': 0.4,  # General variation level for natural look (0-1)
            'character_consistency': 0.7,  # How consistent characters should be (0-1)
            'global_noise': 0.2  # Global noise level for paper texture (0-1)
        }

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize and load the generator
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()
    print(f"Generator model loaded from {generator_path}")

    # Load font
    try:
        font = ImageFont.truetype(ttf_path, 64)
        print(f"Font loaded from {ttf_path}")
    except Exception as e:
        print(f"Error loading font: {e}")
        print("Using default font instead")
        font = ImageFont.load_default()

    # Load paper template
    try:
        paper_template = Image.open(paper_template_path).convert('RGBA')
        print(f"Paper template loaded from {paper_template_path}")
    except Exception as e:
        print(f"Error loading paper template: {e}")
        print("Creating blank template instead")
        paper_template = Image.new('RGBA', (800, 1000), (255, 255, 255, 255))

    # Analyze paper template to find lines
    template_gray = np.array(paper_template.convert('L'))
    template_width, template_height = paper_template.size

    # Detect horizontal lines in the template
    line_positions = []
    threshold = 200  # Threshold for detecting lines

    # Compute the average brightness of each row
    row_brightness = np.mean(template_gray, axis=1)

    # Find rows that are significantly darker (likely to be lines)
    for y in range(1, template_height - 1):
        if (row_brightness[y] < threshold and
                row_brightness[y] < row_brightness[y - 1] and
                row_brightness[y] < row_brightness[y + 1]):
            line_positions.append(y)

    # If no lines were detected, create evenly spaced lines
    if not line_positions:
        print("No lines detected in template, creating evenly spaced lines")
        line_spacing = 30  # Default line spacing
        line_positions = list(range(60, template_height - 60, line_spacing))

    # Calculate the average line spacing
    line_spacing = np.mean(np.diff(line_positions)) if len(line_positions) > 1 else 30
    print(f"Detected {len(line_positions)} lines with average spacing of {line_spacing:.2f} pixels")

    # Split text into lines that fit on the paper
    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + len(current_line) <= max_chars_per_line:
            current_line.append(word)
            current_length += len(word)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)

    if current_line:
        lines.append(' '.join(current_line))

    # Transformation for the generator
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Create a copy of the paper template for the result
    result_img = paper_template.copy()

    # Apply a subtle paper texture to the entire page for realism
    if handwriting_style['global_noise'] > 0:
        paper_noise = np.random.randint(
            0,
            int(7 * handwriting_style['global_noise']),
            (template_height, template_width),
            dtype=np.uint8
        )

        # Convert to PIL, apply noise, and convert back
        paper_pil = result_img.convert('RGBA')
        paper_array = np.array(paper_pil)

        # Only apply to RGB channels, not alpha
        for i in range(3):  # RGB channels
            # Paper texture should be very subtle
            paper_array[:, :, i] = np.clip(
                paper_array[:, :, i].astype(np.int16) - paper_noise // 2 + paper_noise,
                0,
                255
            ).astype(np.uint8)

        result_img = Image.fromarray(paper_array)

    # Process each line of text
    for line_idx, line_text in enumerate(lines):
        if line_idx >= len(line_positions) - 1:
            print(f"Warning: More lines of text than available lines on template. Skipping remaining text.")
            break

        # Get the target y-position for this line (align with the ruled line)
        target_y = line_positions[line_idx]

        # Create character rhythm pattern for this line
        rhythm_pattern = create_handwriting_rhythm(
            len(line_text),
            handwriting_style['rhythm_strength']
        )

        # Calculate global line slant
        base_slant = handwriting_style['slant_angle']

        # Create individual character images and transform them
        char_imgs = []
        for char in line_text:
            if char == ' ':
                # Handle spaces separately - make them narrower
                img_space = Image.new('L', (128, 128), 255)
                img_space_tensor = transform(img_space)
                char_imgs.append(img_space_tensor)
            else:
                img = Image.new('L', (128, 128), 255)
                draw = ImageDraw.Draw(img)

                # Get text size
                try:
                    left, top, right, bottom = font.getbbox(char)
                    w, h = right - left, bottom - top
                except AttributeError:
                    # Fallback for older PIL versions
                    try:
                        w, h = font.getsize(char)
                    except:
                        w, h = 30, 40  # Default size

                draw.text(((128 - w) // 2, (128 - h) // 2), char, font=font, fill=0)
                img_tensor = transform(img)
                char_imgs.append(img_tensor)

        # Generate handwritten versions
        handwritten_chars = []
        with torch.no_grad():
            for img_tensor in char_imgs:
                input_tensor = img_tensor.unsqueeze(0).to(device)
                fake = generator(input_tensor)
                fake_img = (fake.cpu().squeeze().numpy() * 0.5 + 0.5) * 255

                # Apply line removal with style-dependent strength
                if remove_lines and handwriting_style['line_removal_strength'] > 0:
                    effective_threshold = line_threshold - int(15 * handwriting_style['line_removal_strength'])
                    fake_img = smart_line_removal(
                        fake_img,
                        threshold=effective_threshold,
                        min_length=8 + int(4 * handwriting_style['line_removal_strength'])
                    )

                # Apply natural variations based on style
                fake_img = apply_natural_variations(
                    fake_img,
                    variation_level=handwriting_style['variation_level']
                )

                handwritten_chars.append(fake_img)

        # Character boundary detection
        threshold = 245
        char_info = []

        for i, img_array in enumerate(handwritten_chars):
            if line_text[i] == ' ':
                # Handle spaces - make them narrower, but vary width based on rhythm
                space_width = int(20 * rhythm_pattern[i]) if i < len(rhythm_pattern) else 20
                char_info.append({
                    'char': ' ',
                    'cropped': None,
                    'width': space_width,
                    'height': 0,
                    'y_offset': 0,
                    'left_edge': 0,
                    'right_edge': 0
                })
                continue

            # Find the content boundaries
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

                # Apply additional top cropping
                additional_top_crop = int((bottom_edge - top_edge) * top_crop_percent / 100)
                top_edge += additional_top_crop

                # Crop the character image
                if top_edge < bottom_edge and left_edge < right_edge:
                    cropped_char = img_array[top_edge:bottom_edge, left_edge:right_edge]

                    # Convert to PIL Image for further processing
                    cropped_pil = Image.fromarray(cropped_char.astype(np.uint8))

                    # Apply character-specific slant variation
                    char_slant = base_slant
                    if random.random() < 0.7:  # 70% chance to vary from base slant
                        char_slant += random.uniform(-2, 2) * handwriting_style['character_consistency']

                    if char_slant != 0:
                        cropped_pil = apply_slant_variation(cropped_pil, char_slant)

                    # Apply pressure variations
                    alpha_pil = cropped_pil.convert('RGBA')
                    effective_pressure = handwriting_style['pressure_variation'] * rhythm_pattern[i] if i < len(
                        rhythm_pattern) else handwriting_style['pressure_variation']
                    alpha_pil = apply_pressure_variations(alpha_pil, effective_pressure)

                    char_info.append({
                        'char': line_text[i],
                        'cropped': alpha_pil,
                        'width': alpha_pil.width,
                        'height': alpha_pil.height,
                        'y_offset': top_edge,
                        'left_edge': left_edge,
                        'right_edge': right_edge,
                        'baseline_shift': 0,  # Will be updated later
                        'rotation': 0,  # Will be updated later
                        'overlap': int(alpha_pil.width * spacing_factor) if spacing_factor < 0 else 0
                    })
                else:
                    # Use a fallback for characters with detection issues
                    img_pil = Image.fromarray(img_array.astype(np.uint8))
                    char_info.append({
                        'char': line_text[i],
                        'cropped': img_pil.convert('RGBA'),
                        'width': img_pil.width,
                        'height': img_pil.height,
                        'y_offset': 0,
                        'left_edge': 0,
                        'right_edge': img_pil.width,
                        'baseline_shift': 0,
                        'rotation': 0,
                        'overlap': int(img_pil.width * spacing_factor) if spacing_factor < 0 else 0
                    })

                # Apply natural pen movement and consistency adjustments
            char_info = apply_natural_pen_movement(char_info, handwriting_style['movement_variation'])
            char_info = adjust_for_consistency(char_info)

            # Determine line height based on character heights
            avg_height = 0
            count = 0
            for info in char_info:
                if info['char'] != ' ' and info['cropped'] is not None:
                    avg_height += info['cropped'].height
                    count += 1

            if count > 0:
                avg_height = avg_height / count
            else:
                avg_height = line_spacing * 0.7  # Fallback height

            # Paste the characters onto the line
            x_offset = 20  # Starting position from left margin
            for i, info in enumerate(char_info):
                if info['char'] == ' ' or info['cropped'] is None:
                    # Just advance the position for spaces
                    x_offset += info['width']
                    continue

                # Calculate y position (aligned to baseline)
                # Adjust the baseline to the line position
                y_position = target_y - int(info['cropped'].height * 0.7) + info['baseline_shift']

                # Check if we need to apply rotation
                if 'rotation' in info and info['rotation'] != 0:
                    # Create a copy to rotate
                    rotated = info['cropped'].rotate(
                        info['rotation'],
                        resample=Image.BICUBIC,
                        expand=True,
                        fillcolor=(255, 255, 255, 0)  # Transparent fill
                    )

                    # Update positioning to account for size change after rotation
                    width_diff = rotated.width - info['cropped'].width
                    height_diff = rotated.height - info['cropped'].height

                    # Paste the rotated character
                    result_img.paste(
                        rotated,
                        (x_offset - width_diff // 2, y_position - height_diff // 2),
                        rotated
                    )
                else:
                    # Paste regular character
                    result_img.paste(info['cropped'], (x_offset, y_position), info['cropped'])

                # Advance position, accounting for overlap with next character
                if i < len(char_info) - 1 and 'overlap' in info:
                    x_offset += info['width'] - info['overlap']
                else:
                    x_offset += info['width']

                # Check if we've reached the edge of the paper
                if x_offset > template_width - 40:  # 40px margin
                    print(f"Warning: Text exceeded right margin on line {line_idx + 1}")
                    break

            # Save the result
        try:
            result_img.save(output_path)
            print(f"Generated handwritten text saved to {output_path}")
        except Exception as e:
            print(f"Error saving output image: {e}")

        return result_img


# Example usage
if __name__ == "__main__":
    # Example parameters
    generator_path = "handwriting_gan_output/generator.pth"
    ttf_path = "fonts/regular.ttf"
    sample_text = "This is an example of AI-generated handwriting that looks natural and realistic."
    paper_template_path = "templates/lined_paper.png"
    output_path = "output/handwritten_output.png"

    # Define a handwriting style
    casual_style = {
        'line_removal_strength': 0.9,  # How aggressively to remove connecting lines (0-1)
        'pressure_variation': 0.5,  # Variation in pen pressure (0-1)
        'movement_variation': 0.6,  # Variation in pen movement/baseline (0-1)
        'slant_angle': 3,  # Overall slant of the handwriting
        'rhythm_strength': 0.6,  # Strength of natural writing rhythm (0-1)
        'variation_level': 0.5,  # General variation level for natural look (0-1)
        'character_consistency': 0.7,  # How consistent characters should be (0-1)
        'global_noise': 0.3  # Global noise level for paper texture (0-1)
    }

    # Generate the handwritten text
    generate_text_on_lined_paper(
        generator_path,
        ttf_path,
        sample_text,
        paper_template_path,
        output_path,
        spacing_factor=-0.2,  # Small overlap between characters
        handwriting_style=casual_style
    )
