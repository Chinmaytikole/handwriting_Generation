import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import numpy as np
from torchvision import transforms
from scipy import ndimage
import re


# Define the Generator network
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


# Create a more flexible generator that handles checkpoint format differences
class FlexibleGenerator(nn.Module):
    def _init_(self):
        super(FlexibleGenerator, self)._init_()
        # Define attributes with the names that match what's being referenced
        self.down1_conv = nn.Conv2d(1, 64, 4, 2, 1)
        self.down1_lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.down2_conv = nn.Conv2d(64, 128, 4, 2, 1)
        self.down2_bn = nn.BatchNorm2d(128)
        self.down2_lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.down3_conv = nn.Conv2d(128, 256, 4, 2, 1)
        self.down3_bn = nn.BatchNorm2d(256)
        self.down3_lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.down4_conv = nn.Conv2d(256, 512, 4, 2, 1)
        self.down4_bn = nn.BatchNorm2d(512)
        self.down4_lrelu = nn.LeakyReLU(0.2, inplace=True)

        # Upsampling layers
        self.up1_conv = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.up1_bn = nn.BatchNorm2d(256)
        self.up1_relu = nn.ReLU(inplace=True)

        self.up2_conv = nn.ConvTranspose2d(512, 128, 4, 2, 1)
        self.up2_bn = nn.BatchNorm2d(128)
        self.up2_relu = nn.ReLU(inplace=True)

        self.up3_conv = nn.ConvTranspose2d(256, 64, 4, 2, 1)
        self.up3_bn = nn.BatchNorm2d(64)
        self.up3_relu = nn.ReLU(inplace=True)

        self.up4_conv = nn.ConvTranspose2d(128, 1, 4, 2, 1)
        self.up4_tanh = nn.Tanh()

    def forward(self, x):
        # Downsampling path
        d1_conv = self.down1_conv(x)
        d1 = self.down1_lrelu(d1_conv)

        d2_conv = self.down2_conv(d1)
        d2_bn = self.down2_bn(d2_conv)
        d2 = self.down2_lrelu(d2_bn)

        d3_conv = self.down3_conv(d2)
        d3_bn = self.down3_bn(d3_conv)
        d3 = self.down3_lrelu(d3_bn)

        d4_conv = self.down4_conv(d3)
        d4_bn = self.down4_bn(d4_conv)
        d4 = self.down4_lrelu(d4_bn)

        # Upsampling path
        u1_conv = self.up1_conv(d4)
        u1_bn = self.up1_bn(u1_conv)
        u1 = self.up1_relu(u1_bn)

        u2_input = torch.cat([u1, d3], 1)
        u2_conv = self.up2_conv(u2_input)
        u2_bn = self.up2_bn(u2_conv)
        u2 = self.up2_relu(u2_bn)

        u3_input = torch.cat([u2, d2], 1)
        u3_conv = self.up3_conv(u3_input)
        u3_bn = self.up3_bn(u3_conv)
        u3 = self.up3_relu(u3_bn)

        u4_input = torch.cat([u3, d1], 1)
        u4_conv = self.up4_conv(u4_input)
        u4 = self.up4_tanh(u4_conv)

        return u4


# Advanced image cleaning function
def advanced_clean_image(img_array, threshold=225):
    """
    Advanced image cleaning to significantly improve output quality

    Args:
        img_array: NumPy array of grayscale image
        threshold: Pixel value threshold for content detection

    Returns:
        Cleaned image array
    """
    # Create a copy to work with
    result = img_array.copy()

    # Step 1: Initial thresholding to identify clear content
    binary = (result < threshold).astype(np.uint8)

    # Step 2: Label connected components
    labeled, num_features = ndimage.label(binary)

    # Step 3: Remove very small components (likely noise)
    component_sizes = np.bincount(labeled.ravel())
    too_small = component_sizes < 5
    too_small_mask = too_small[labeled]
    result[too_small_mask] = 255  # Set to white

    # Step 4: Enhance contrast on the main content
    # First, identify content regions
    binary = (result < threshold).astype(np.uint8)
    content_mask = binary > 0

    if np.any(content_mask):
        # Apply contrast stretching to content only
        min_val = np.percentile(result[content_mask], 5)
        max_val = np.percentile(result[content_mask], 95)

        if max_val > min_val:
            # Calculate intensity range to improve contrast
            content_range = result[content_mask]
            content_range = np.clip(((content_range - min_val) / (max_val - min_val)) * 220, 0, 220)
            result[content_mask] = content_range

            # Make content darker (more black)
            result[content_mask] = np.clip(result[content_mask] * 0.8, 0, 255)

    # Step 5: Remove horizontal connecting lines
    # First create a binary image
    binary = (result < threshold).astype(np.uint8)

    # Identify potential horizontal lines
    # Horizontal structure element
    h_struct = np.ones((1, 5))
    h_eroded = ndimage.binary_erosion(binary, h_struct)
    h_reconstructed = ndimage.binary_dilation(h_eroded, h_struct)

    # Potential lines are where we have horizontal erosion/dilation matches
    # but not much vertical structure
    v_struct = np.ones((5, 1))
    v_eroded = ndimage.binary_erosion(binary, v_struct)

    # Lines are horizontal structures with minimal vertical components
    potential_lines = h_reconstructed & ~v_eroded

    # Remove these potential connecting lines
    result[potential_lines] = 255

    # Step 6: Final cleanup
    # Median filter to remove remaining noise while preserving edges
    result = ndimage.median_filter(result, size=2)

    return result


# Enhanced function to generate high-quality handwritten text
def generate_handwritten_text(generator_path, ttf_path, text, output_path,
                              char_spacing=-0.25,  # Increased spacing between characters
                              word_spacing=20,  # More space between words
                              contrast_factor=1.5,  # Enhance contrast
                              darkness_factor=0.7,  # Make text darker
                              content_threshold=220):  # Lower threshold for better content detection

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize and load the generator
    generator = FlexibleGenerator().to(device)
    generator.load_state_dict_from_checkpoint(generator_path, map_location=device)
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

    # Transformation
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Process each character
    char_imgs = []
    char_is_space = []  # Track which characters are spaces

    for char in text:
        if char == ' ':
            # Just track this as a space - no need to process
            char_imgs.append(None)
            char_is_space.append(True)
        else:
            # Create character image
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

            # Center the character
            draw.text(((128 - w) // 2, (128 - h) // 2), char, font=font, fill=0)
            img_tensor = transform(img)
            char_imgs.append(img_tensor)
            char_is_space.append(False)

    # Generate handwritten versions
    handwritten_chars = []

    with torch.no_grad():
        for i, is_space in enumerate(char_is_space):
            if is_space:
                # Add a placeholder for spaces
                handwritten_chars.append(None)
            else:
                # Process actual character
                input_tensor = char_imgs[i].unsqueeze(0).to(device)
                fake = generator(input_tensor)
                fake_img = (fake.cpu().squeeze().numpy() * 0.5 + 0.5) * 255

                # Apply advanced cleaning function
                fake_img = advanced_clean_image(fake_img, threshold=content_threshold)

                handwritten_chars.append(fake_img)

    # Character boundary detection with improved algorithm
    char_info = []

    for i, (img_array, is_space) in enumerate(zip(handwritten_chars, char_is_space)):
        if is_space:
            # Handle spaces properly - explicit width
            char_info.append({
                'char': ' ',
                'cropped': None,
                'width': word_spacing,  # Explicit word spacing
                'height': 0,
                'y_offset': 0
            })
            continue

        # Create binary image for content detection
        binary = (img_array < content_threshold).astype(np.uint8)

        # Apply morphological closing to connect nearby components
        binary = ndimage.binary_closing(binary, structure=np.ones((2, 2)))

        # Find content boundaries
        rows_with_content = np.any(binary, axis=1)
        cols_with_content = np.any(binary, axis=0)

        if np.any(rows_with_content) and np.any(cols_with_content):
            # Find the content boundaries
            top_edge = np.argmax(rows_with_content)
            bottom_edge = img_array.shape[0] - np.argmax(rows_with_content[::-1])
            left_edge = np.argmax(cols_with_content)
            right_edge = img_array.shape[1] - np.argmax(cols_with_content[::-1])

            # Add padding around character
            padding = 2
            top_edge = max(0, top_edge - padding)
            bottom_edge = min(img_array.shape[0], bottom_edge + padding)
            left_edge = max(0, left_edge - padding)
            right_edge = min(img_array.shape[1], right_edge + padding)

            # Create a copy of the image region
            cropped = img_array[top_edge:bottom_edge, left_edge:right_edge].copy()

            # Create an alpha mask where white becomes transparent
            alpha = np.where(cropped < content_threshold, 255, 0).astype(np.uint8)  # Invert logic for better results

            # Create an RGBA image
            rgba = np.zeros((cropped.shape[0], cropped.shape[1], 4), dtype=np.uint8)

            # Set the RGB values - make text black for better visibility
            for c in range(3):  # RGB channels
                rgba[:, :, c] = np.where(cropped < content_threshold, 0, 255)

            # Set the alpha channel - fully opaque for text pixels
            rgba[:, :, 3] = alpha

            # Convert to PIL Image
            cropped_img = Image.fromarray(rgba)

            # Enhance contrast to make text clearer
            enhancer = ImageEnhance.Contrast(cropped_img)
            cropped_img = enhancer.enhance(contrast_factor)

            # Sharpen the image to make characters more defined
            cropped_img = cropped_img.filter(ImageFilter.SHARPEN)

            char_info.append({
                'char': text[i],
                'cropped': cropped_img,
                'width': cropped.shape[1],
                'height': cropped.shape[0],
                'y_offset': top_edge
            })
        else:
            # Empty character (but not a space)
            char_info.append({
                'char': text[i],
                'cropped': None,
                'width': 5,
                'height': 0,
                'y_offset': 0
            })

    # Calculate the overlap amounts for kerning
    for i in range(1, len(char_info)):
        # Skip if either character is a space or empty
        if char_info[i]['cropped'] is None or char_info[i - 1]['cropped'] is None:
            continue

        # Check if we're starting a new word (previous char was space)
        if i > 0 and char_info[i - 1]['char'] == ' ':
            char_info[i]['overlap'] = 0  # No overlap at start of words
        else:
            # Adjust overlap based on character pairs
            prev_char = char_info[i - 1]['char']
            curr_char = char_info[i]['char']

            # Customize spacing for certain character pairs
            if prev_char in 'rvw' and curr_char in 'rvw':
                # More spacing between similar slanted characters
                char_info[i]['overlap'] = int(char_info[i]['width'] * abs(char_spacing) * 0.8)
            else:
                # Normal character overlap
                char_info[i]['overlap'] = int(char_info[i]['width'] * abs(char_spacing))

    # Calculate total width considering overlaps
    total_width = 0
    for i, info in enumerate(char_info):
        if i > 0 and 'overlap' in info:
            total_width += info['width'] - info['overlap']
        else:
            total_width += info['width']

    # Add margin
    total_width += 20

    # Find max height
    max_height = max((info['height'] for info in char_info if info['height'] > 0), default=50)

    # Create a new image with RGB mode for better quality
    result_img = Image.new('RGB', (total_width, max_height + 30), (255, 255, 255))  # White background

    # Paste characters with improved positioning
    x_offset = 10  # Left margin
    baseline = max_height * 0.7  # Lower baseline for better alignment

    # First pass - calculate average height for better baseline alignment
    valid_heights = [info['height'] for info in char_info if info['height'] > 0]
    if valid_heights:
        avg_height = sum(valid_heights) / len(valid_heights)
        baseline = max_height - (avg_height * 0.5)  # Adjust baseline based on average character height

    for i, info in enumerate(char_info):
        if info['cropped'] is not None:
            # Apply kerning/overlap adjustment for characters after the first one
            if i > 0 and 'overlap' in info:
                x_offset -= info['overlap']

            # Calculate vertical position with baseline alignment
            y_offset = int(baseline - info['height'] * 0.6)  # Lower placement

            # Additional adjustments for specific characters
            if info['char'] in 'gjpqy':  # Characters with descenders
                y_offset += 4  # Move down
            elif info['char'] in 'bdfhkl':  # Tall characters
                y_offset -= 2  # Move up slightly

            # Paste the character - use a mask for clean transparency
            result_img.paste((0, 0, 0), (x_offset, y_offset, x_offset + info['width'], y_offset + info['height']),
                             mask=info['cropped'])

        # Update x_offset
        x_offset += info['width']

    # Final enhancement pass for overall image
    # Convert back to PIL Image for post-processing
    enhancer = ImageEnhance.Contrast(result_img)
    result_img = enhancer.enhance(1.2)  # Slightly boost contrast

    # Sharpen the final image
    result_img = result_img.filter(ImageFilter.SHARPEN)

    # Save the result
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    result_img.save(output_path)
    print(f"Generated handwritten text saved to {output_path}")

    return result_img


# Main function
if __name__ == "__main__":
    # Default parameters
    DEFAULT_GENERATOR_PATH = "handwriting_gan_output/generator.pth"
    DEFAULT_FONT_PATH = "Myfont-Regular.ttf"
    DEFAULT_TEXT = "Hello World"
    DEFAULT_OUTPUT_PATH = "handwritten_output.png"
    DEFAULT_CHAR_SPACING = -0.25  # Less negative for wider spacing
    DEFAULT_WORD_SPACING = 20  # More space between words
    DEFAULT_CONTRAST = 1.5  # Higher contrast
    DEFAULT_DARKNESS = 0.7  # Darker text
    DEFAULT_THRESHOLD = 220  # Lower threshold for better content detection

    import sys

    # Simple command line parsing
    generator_path = DEFAULT_GENERATOR_PATH
    font_path = DEFAULT_FONT_PATH
    text = DEFAULT_TEXT
    output_path = DEFAULT_OUTPUT_PATH
    char_spacing = DEFAULT_CHAR_SPACING
    word_spacing = DEFAULT_WORD_SPACING
    contrast = DEFAULT_CONTRAST
    darkness = DEFAULT_DARKNESS
    threshold = DEFAULT_THRESHOLD

    # Check if any arguments were provided
    if len(sys.argv) > 1:
        # Parse arguments
        args = sys.argv[1:]
        i = 0
        while i < len(args):
            if args[i] == "--generator" and i + 1 < len(args):
                generator_path = args[i + 1]
                i += 2
            elif args[i] == "--font" and i + 1 < len(args):
                font_path = args[i + 1]
                i += 2
            elif args[i] == "--text" and i + 1 < len(args):
                text = args[i + 1]
                i += 2
            elif args[i] == "--output" and i + 1 < len(args):
                output_path = args[i + 1]
                i += 2
            elif args[i] == "--char-spacing" and i + 1 < len(args):
                char_spacing = float(args[i + 1])
                i += 2
            elif args[i] == "--word-spacing" and i + 1 < len(args):
                word_spacing = int(args[i + 1])
                i += 2
            elif args[i] == "--contrast" and i + 1 < len(args):
                contrast = float(args[i + 1])
                i += 2
            elif args[i] == "--darkness" and i + 1 < len(args):
                darkness = float(args[i + 1])
                i += 2
            elif args[i] == "--threshold" and i + 1 < len(args):
                threshold = int(args[i + 1])
                i += 2
            else:
                # If no recognized flag, assume it's the text
                text = args[i]
                i += 1

    # Display configuration
    print(f"Configuration:")
    print(f"  Generator path: {generator_path}")
    print(f"  Font path: {font_path}")
    print(f"  Text: '{text}'")
    print(f"  Output path: {output_path}")
    print(f"  Character spacing: {char_spacing}")
    print(f"  Word spacing: {word_spacing}px")
    print(f"  Contrast: {contrast}")
    print(f"  Darkness: {darkness}")
    print(f"  Threshold: {threshold}")

    # Generate handwritten text
    try:
        result = generate_handwritten_text(
            generator_path=generator_path,
            ttf_path=font_path,
            text=text,
            output_path=output_path,
            char_spacing=char_spacing,
            word_spacing=word_spacing,
            contrast_factor=contrast,
            darkness_factor=darkness,
            content_threshold=threshold
        )
        print("Handwritten text generation completed successfully.")
    except Exception as e:
        print(f"Error generating handwritten text: {e}")
        sys.exit(1)