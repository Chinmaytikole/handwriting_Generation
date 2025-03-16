import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from torchvision import transforms


# Define the Generator and Discriminator networks
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


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # Input: 2 channels (input + condition) x 128 x 128
            nn.Conv2d(2, 64, 4, 2, 1),  # -> 64 x 64 x 64
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),  # -> 128 x 32 x 32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),  # -> 256 x 16 x 16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),  # -> 512 x 8 x 8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 1),  # -> 1 x 7 x 7
            nn.Sigmoid()
        )

    def forward(self, x, y):
        # Concatenate input and condition
        input_tensor = torch.cat([x, y], 1)
        return self.model(input_tensor)


# Dataset for font samples
class FontDataset(Dataset):
    def __init__(self, ttf_path, real_samples_dir, transform=None):
        self.font = ImageFont.truetype(ttf_path, 64)
        self.transform = transform
        self.characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

        # Load real handwriting samples if provided
        self.real_samples = []
        if real_samples_dir and os.path.exists(real_samples_dir):
            for file in os.listdir(real_samples_dir):
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(real_samples_dir, file)
                    img = Image.open(img_path).convert('L')
                    if self.transform:
                        img = self.transform(img)
                    self.real_samples.append(img)

    def __len__(self):
        return len(self.characters)

    def __getitem__(self, idx):
        char = self.characters[idx]

        # Create font image
        img = Image.new('L', (128, 128), 255)
        draw = ImageDraw.Draw(img)

        # Get text size using getbbox() instead of textsize()
        font = self.font
        left, top, right, bottom = font.getbbox(char)
        w, h = right - left, bottom - top

        # Draw text centered
        draw.text(((128 - w) // 2, (128 - h) // 2), char, font=font, fill=0)

        # Apply transform
        if self.transform:
            img_tensor = self.transform(img)

        # Return real sample if available, otherwise use synthetic as both input and target
        if idx < len(self.real_samples):
            real_sample = self.real_samples[idx]
            return img_tensor, real_sample
        else:
            # Add slight randomness to create "target" with variation
            img_variation = img.copy()
            draw_var = ImageDraw.Draw(img_variation)

            # Slight position variation
            offset_x = np.random.randint(-3, 4)
            offset_y = np.random.randint(-3, 4)
            draw_var.text(((128 - w) // 2 + offset_x, (128 - h) // 2 + offset_y), char, font=font, fill=0)
            img_variation_tensor = self.transform(img_variation)

            return img_tensor, img_variation_tensor


# Training function
def train_handwriting_gan(ttf_path, real_samples_dir=None, output_dir="output", num_epochs=100):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])

    # Create dataset and dataloader
    dataset = FontDataset(ttf_path, real_samples_dir, transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Initialize networks
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Loss functions
    criterion_gan = nn.BCELoss()
    criterion_pixelwise = nn.L1Loss()

    # Lambda for L1 loss
    lambda_pixel = 100

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Training loop
    for epoch in range(num_epochs):
        for i, (font_imgs, real_imgs) in enumerate(dataloader):
            batch_size = font_imgs.size(0)

            # Configure input
            real_A = font_imgs.to(device)  # Font rendered image
            real_B = real_imgs.to(device)  # Target "real" handwriting

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Generate fake handwriting
            fake_B = generator(real_A)

            # GAN loss
            pred_fake = discriminator(fake_B, real_A)

            # Create valid tensor with the same size as pred_fake
            # Get the actual sizes from the discriminator output
            target_h, target_w = pred_fake.size(2), pred_fake.size(3)
            valid = torch.ones(batch_size, 1, target_h, target_w).to(device)
            fake = torch.zeros(batch_size, 1, target_h, target_w).to(device)

            loss_GAN = criterion_gan(pred_fake, valid)

            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(fake_B, real_B)

            # Total loss
            loss_G = loss_GAN + lambda_pixel * loss_pixel

            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Real loss
            pred_real = discriminator(real_B, real_A)
            loss_real = criterion_gan(pred_real, valid)

            # Fake loss
            pred_fake = discriminator(fake_B.detach(), real_A)
            loss_fake = criterion_gan(pred_fake, fake)

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)

            loss_D.backward()
            optimizer_D.step()

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"[Epoch {epoch + 1}/{num_epochs}] [D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]")

            # Save sample images
            with torch.no_grad():
                fake_samples = generator(real_A)

                # Convert tensors to PIL images
                for j in range(min(3, batch_size)):
                    # Convert to PIL images
                    font_img = (real_A[j].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255
                    fake_img = (fake_samples[j].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255
                    real_img = (real_B[j].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255

                    # Create a comparison image
                    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                    axs[0].imshow(font_img.squeeze(), cmap='gray')
                    axs[0].set_title('Font Render')
                    axs[0].axis('off')

                    axs[1].imshow(fake_img.squeeze(), cmap='gray')
                    axs[1].set_title('GAN Output')
                    axs[1].axis('off')

                    axs[2].imshow(real_img.squeeze(), cmap='gray')
                    axs[2].set_title('Target')
                    axs[2].axis('off')

                    plt.savefig(f"{output_dir}/sample_epoch{epoch + 1}_idx{j}.png")
                    plt.close()

    # Save the trained model
    torch.save(generator.state_dict(), f"{output_dir}/generator.pth")
    torch.save(discriminator.state_dict(), f"{output_dir}/discriminator.pth")

    return generator, discriminator


# Function to apply the model to new text
def generate_handwritten_text(generator, ttf_path, text, output_path, spacing_factor=0.1):
    """
    Generate handwritten text with improved spacing between letters

    Args:
        generator: The trained generator model
        ttf_path: Path to the font file
        text: Text to generate as handwriting
        output_path: Where to save the output image
        spacing_factor: Controls the spacing between letters (lower = less space)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.eval()

    # Load font
    font = ImageFont.truetype(ttf_path, 64)

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
            img = Image.new('L', (128, 128), 255)
            draw = ImageDraw.Draw(img)

            # Get text size using getbbox() instead of textsize()
            left, top, right, bottom = font.getbbox(char)
            w, h = right - left, bottom - top

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
            handwritten_chars.append(fake_img)

    # Calculate bounding boxes for each character to remove excess whitespace
    char_bboxes = []
    for img_array in handwritten_chars:
        # Convert to binary: 0 for white (threshold at 240), 1 for content
        binary = (img_array < 240).astype(np.uint8)

        # Find content boundaries (non-white pixels)
        rows_with_content = np.any(binary, axis=1)
        cols_with_content = np.any(binary, axis=0)

        if np.any(rows_with_content) and np.any(cols_with_content):
            # Find top, bottom, left, right boundaries
            top = np.argmax(rows_with_content)
            bottom = img_array.shape[0] - np.argmax(rows_with_content[::-1])
            left = np.argmax(cols_with_content)
            right = img_array.shape[1] - np.argmax(cols_with_content[::-1])

            # Add minimal padding (1-2 pixels) to avoid cutting off parts
            padding = 2
            top = max(0, top - padding)
            bottom = min(img_array.shape[0], bottom + padding)
            left = max(0, left - padding)
            right = min(img_array.shape[1], right + padding)

            # Store bounding box
            char_bboxes.append((top, bottom, left, right))
        else:
            # For spaces or empty characters
            width = int(img_array.shape[1] * 0.3)  # Make spaces narrower
            char_bboxes.append((0, img_array.shape[0], 0, width))

    # Calculate dimensions for combined image
    total_width = 0
    max_height = 0

    # First pass: calculate total width with spacing_factor
    for i, (top, bottom, left, right) in enumerate(char_bboxes):
        char_width = right - left

        if text[i % len(text)] == ' ':
            # Handle spaces differently - make them fixed width
            space_width = int(char_width * 0.06)  # Narrower spaces
            total_width += space_width
        else:
            # Add space between characters based on spacing_factor
            added_space = int(char_width * spacing_factor)
            total_width += char_width + added_space

        char_height = bottom - top
        max_height = max(max_height, char_height)

    # Ensure minimum height
    max_height = max(max_height, 64)

    # Create blank image with calculated dimensions
    combined_img = Image.new('L', (total_width, max_height), 255)

    # Second pass: paste cropped characters with adjusted spacing
    x_offset = 0
    for i, img_array in enumerate(handwritten_chars):
        top, bottom, left, right = char_bboxes[i]

        # Crop character to its bounding box
        char_height = bottom - top
        char_width = right - left

        if text[i % len(text)] == ' ':
            # For spaces, just add space without pasting anything
            x_offset += int(char_width * 0.6)
            continue

        # Crop the character image to its bounding box
        char_img = Image.fromarray(img_array.astype(np.uint8)).crop((left, top, right, bottom))

        # Calculate vertical centeringA
        y_offset = (max_height - char_height) // 2

        # Paste at current offset with vertical centering
        combined_img.paste(char_img, (x_offset, y_offset))

        # Update offset for next character
        added_space = int(char_width * spacing_factor)
        x_offset += char_width + added_space

    # Save combined image
    combined_img.save(output_path)
    print(f"Generated handwritten text saved to {output_path}")

    return combined_img


def remove_connecting_lines(img_array, threshold=240, min_length=5, max_thickness=5):
    """
    Enhanced version to more aggressively remove connecting lines between characters
    """
    height, width = img_array.shape
    result = img_array.copy()

    # Make binary image for line detection
    binary = (img_array < threshold).astype(np.uint8)

    # Scan through rows looking for horizontal lines
    for y in range(height):
        # Count continuous runs of dark pixels in this row
        run_lengths = []
        in_run = False
        run_length = 0
        run_start = 0

        for x in range(width):
            if binary[y, x] == 1:  # Dark pixel
                if not in_run:
                    in_run = True
                    run_length = 1
                    run_start = x
                else:
                    run_length += 1
            else:  # Light pixel
                if in_run:
                    in_run = False
                    if run_length >= min_length:
                        run_lengths.append((run_start, run_length))
                    run_length = 0

        # Don't forget the last run
        if in_run and run_length >= min_length:
            run_lengths.append((run_start, run_length))

        # For each horizontal run, check if it's likely a connecting line
        for run_start, run_length in run_lengths:
            # Check thickness by looking at rows above and below
            is_thin_line = True

            # Check rows above and below to see if this is an isolated line
            for t in range(1, max_thickness + 1):
                if y - t >= 0:
                    # Check if pixels above are mostly white
                    pixels_above = binary[y - t, run_start:run_start + run_length]
                    if np.sum(pixels_above) > run_length * 0.3:  # More than 30% dark pixels above
                        is_thin_line = False
                        break

                if y + t < height:
                    # Check if pixels below are mostly white
                    pixels_below = binary[y + t, run_start:run_start + run_length]
                    if np.sum(pixels_below) > run_length * 0.3:  # More than 30% dark pixels below
                        is_thin_line = False
                        break

            # If it's a thin isolated line, remove it
            if is_thin_line:
                result[y, run_start:run_start + run_length] = 255

    return result

# Main function to run the complete process
def main(ttf_path, text_to_generate, real_samples_dir=None, num_epochs=100, spacing_factor=0.2):
    # Create output directory
    output_dir = "handwriting_gan_output"
    os.makedirs(output_dir, exist_ok=True)

    # Train the model
    print("Training the GAN model...")
    generator, discriminator = train_handwriting_gan(ttf_path, real_samples_dir, output_dir, num_epochs)

    # Generate handwritten text
    output_path = f"{output_dir}/{text_to_generate.replace(' ', '_')}.png"
    print(f"Generating handwritten text: '{text_to_generate}'")
    generate_handwritten_text(generator, ttf_path, text_to_generate, output_path, spacing_factor)
    print(f"Complete! Check {output_path} for the result.")

    return output_path

    # Example usage


if __name__ == "__main__":
    ttf_path = "Myfont-Regular.ttf"  # Put your TTF file path here
    text_to_generate = "Hello World"
    real_samples_dir = None  # Optional: folder with real handwriting samples
    spacing_factor = 0.8  # Adjust this value to control spacing (lower = less space)

    main(ttf_path, text_to_generate, real_samples_dir, num_epochs=100, spacing_factor=spacing_factor)