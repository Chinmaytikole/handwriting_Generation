import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import matplotlib.pyplot as plt
from torchvision import transforms
from scipy import ndimage


# Define the Generator with improved architecture
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Downsampling layers with more filters
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
        # Added a deeper bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Upsampling layers with skip connections
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)  # Added dropout for better generalization
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Final layer with sigmoid activation for cleaner background
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 1, 4, 2, 1),
            nn.Sigmoid()  # Changed from Tanh to Sigmoid for cleaner binary output
        )

        # Add refinement layer to clean up artifacts
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

        # Add bottleneck
        b = self.bottleneck(d4)

        u1 = self.up1(b)
        u2 = self.up2(torch.cat([u1, d3], 1))
        u3 = self.up3(torch.cat([u2, d2], 1))
        u4 = self.up4(torch.cat([u3, d1], 1))

        # Refinement pass to clean up artifacts
        refined = self.refine(u4)
        return refined


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


# Helper function for perspective transform in data augmentation
def find_coeffs(pa, pb):
    """Find coefficients for perspective transformation"""
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = np.matrix(matrix, dtype=np.float32)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


class FontDataset(Dataset):
    def __init__(self, ttf_path, real_samples_dir, transform=None, augment=True):
        self.font = ImageFont.truetype(ttf_path, 64)
        self.transform = transform
        self.augment = augment
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
        return len(self.characters) * (3 if self.augment else 1)  # Multiple augmentations per character

    def __getitem__(self, idx):
        char_idx = idx % len(self.characters)
        char = self.characters[char_idx]

        # Create font image with white background (255)
        img = Image.new('L', (128, 128), 255)
        draw = ImageDraw.Draw(img)

        # Get text size using getbbox()
        font = self.font
        left, top, right, bottom = font.getbbox(char)
        w, h = right - left, bottom - top

        # Draw text centered
        draw.text(((128 - w) // 2, (128 - h) // 2), char, font=font, fill=0)

        # Apply data augmentation based on augment index
        aug_idx = idx // len(self.characters) if self.augment else 0

        if self.augment and aug_idx > 0:
            # Apply various slight transformations to increase variety
            img_aug = img.copy()

            if aug_idx == 1:
                # Slight rotation
                angle = np.random.uniform(-5, 5)
                img_aug = img_aug.rotate(angle, resample=Image.BICUBIC, fillcolor=255)
            elif aug_idx == 2:
                # Slight perspective transformation
                width, height = img.size
                coeffs = find_coeffs(
                    [(0, 0), (width, 0), (width, height), (0, height)],
                    [(np.random.uniform(0, 10), np.random.uniform(0, 10)),
                     (width - np.random.uniform(0, 10), np.random.uniform(0, 10)),
                     (width - np.random.uniform(0, 10), height - np.random.uniform(0, 10)),
                     (np.random.uniform(0, 10), height - np.random.uniform(0, 10))]
                )
                img_aug = img.transform((width, height), Image.PERSPECTIVE, coeffs,
                                        Image.BICUBIC, fillcolor=255)

            img = img_aug

        # Apply transform
        if self.transform:
            img_tensor = self.transform(img)

        # Return real sample if available, otherwise use synthetic as both input and target
        if char_idx < len(self.real_samples):
            real_sample = self.real_samples[char_idx]
            return img_tensor, real_sample
        else:
            # Create "target" with variation for more realistic handwriting
            img_variation = img.copy()
            draw_var = ImageDraw.Draw(img_variation)

            # Apply slight noise and variation to simulate handwriting
            offset_x = np.random.randint(-3, 4)
            offset_y = np.random.randint(-3, 4)

            # Use a thicker pen for target
            font_size = 64 + np.random.randint(-4, 5)
            target_font = ImageFont.truetype(self.font.path, font_size)

            # Draw with slight position variation
            draw_var.text(((128 - w) // 2 + offset_x, (128 - h) // 2 + offset_y),
                          char, font=target_font, fill=0)

            # Apply a slight blur for more natural appearance
            img_variation = img_variation.filter(ImageFilter.GaussianBlur(radius=0.5))

            img_variation_tensor = self.transform(img_variation)
            return img_tensor, img_variation_tensor


# Function to clean up background noise
def clean_background(image, threshold=240):
    """
    Clean up background noise in the image
    Args:
        image: numpy array of image
        threshold: pixel values above this will be set to 255 (white)
    Returns:
        cleaned image
    """
    cleaned = image.copy()
    cleaned[cleaned > threshold] = 255

    # Further cleanup by removing isolated pixels
    if len(cleaned.shape) == 2:  # For grayscale images
        # Create a binary image
        binary = (cleaned < 255).astype(np.uint8)

        # Remove small objects (noise)
        labeled_array, num_features = ndimage.label(binary)
        component_sizes = np.bincount(labeled_array.ravel())
        # Set small components to background (white)
        small_size = 5  # Minimum size of objects to keep
        too_small = component_sizes < small_size
        too_small_mask = too_small[labeled_array]
        cleaned[too_small_mask] = 255

    return cleaned


# Training function with improved parameters
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

    # Create dataset and dataloader with augmentation
    dataset = FontDataset(ttf_path, real_samples_dir, transform, augment=True)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Initialize networks
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Loss functions
    criterion_gan = nn.BCELoss()
    criterion_pixelwise = nn.L1Loss()

    # Add structural similarity loss for better pattern matching
    criterion_structure = nn.MSELoss()  # Could be replaced with SSIM if available

    # Lambda for losses
    lambda_pixel = 100
    lambda_structure = 10

    # Optimizers with adjusted learning rates
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

    # Learning rate schedulers
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.5)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.5)

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss_G = 0
        epoch_loss_D = 0
        epoch_samples = 0

        for i, (font_imgs, real_imgs) in enumerate(dataloader):
            batch_size = font_imgs.size(0)
            epoch_samples += batch_size

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

            # Get sizes for valid tensors
            target_h, target_w = pred_fake.size(2), pred_fake.size(3)
            valid = torch.ones(batch_size, 1, target_h, target_w).to(device)
            fake = torch.zeros(batch_size, 1, target_h, target_w).to(device)

            loss_GAN = criterion_gan(pred_fake, valid)

            # Pixel-wise loss (L1 loss)
            loss_pixel = criterion_pixelwise(fake_B, real_B)

            # Structural similarity loss for better pattern matching
            loss_structure = criterion_structure(fake_B, real_B)

            # Total generator loss
            loss_G = loss_GAN + lambda_pixel * loss_pixel + lambda_structure * loss_structure

            loss_G.backward()
            optimizer_G.step()

            # -----------------
            #  Train Discriminator
            # -----------------
            optimizer_D.zero_grad()

            # Real loss
            pred_real = discriminator(real_B, real_A)
            loss_real = criterion_gan(pred_real, valid)

            # Fake loss
            pred_fake = discriminator(fake_B.detach(), real_A)
            loss_fake = criterion_gan(pred_fake, fake)

            # Total discriminator loss
            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            optimizer_D.step()

            # Update epoch losses
            epoch_loss_G += loss_G.item() * batch_size
            epoch_loss_D += loss_D.item() * batch_size

            # Print progress
            if i % 10 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}, "
                      f"pixel: {loss_pixel.item():.4f}, structure: {loss_structure.item():.4f}]")

            # Update learning rates
        scheduler_G.step()
        scheduler_D.step()

        # Calculate average epoch losses
        avg_loss_G = epoch_loss_G / epoch_samples
        avg_loss_D = epoch_loss_D / epoch_samples

        print(f"[Epoch {epoch}/{num_epochs}] [Avg G loss: {avg_loss_G:.4f}] [Avg D loss: {avg_loss_D:.4f}]")

        # Save model checkpoints
        # Only save the final model after training is complete
        if epoch == num_epochs - 1:
            torch.save(generator.state_dict(), os.path.join(output_dir, "generator_final.pth"))
            torch.save(discriminator.state_dict(), os.path.join(output_dir, "discriminator_final.pth"))
        # Save sample images
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            with torch.no_grad():
                # Get a batch of test images
                test_A = next(iter(dataloader))[0][:4].to(device)
                test_B = next(iter(dataloader))[1][:4].to(device)

                # Generate output
                fake_B = generator(test_A)

                # Convert to numpy arrays
                test_A = (test_A * 0.5 + 0.5).cpu().numpy()
                test_B = (test_B * 0.5 + 0.5).cpu().numpy()
                fake_B = (fake_B * 0.5 + 0.5).cpu().numpy()

                # Clean up background noise
                for i in range(fake_B.shape[0]):
                    fake_B[i, 0] = clean_background(fake_B[i, 0] * 255) / 255

                # Create figure
                fig, axs = plt.subplots(3, 4, figsize=(12, 9))

                # Plot images
                for i in range(4):
                    # Font image
                    axs[0, i].imshow(test_A[i, 0], cmap='gray')
                    axs[0, i].set_title('Font')
                    axs[0, i].axis('off')

                    # Real handwriting
                    axs[1, i].imshow(test_B[i, 0], cmap='gray')
                    axs[1, i].set_title('Target')
                    axs[1, i].axis('off')

                    # Generated handwriting
                    axs[2, i].imshow(fake_B[i, 0], cmap='gray')
                    axs[2, i].set_title('Generated')
                    axs[2, i].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"samples_epoch_{epoch}.png"))
                plt.close()

# Function to generate handwriting from a saved model
def generate_handwriting(input_text, font_path, model_path, output_dir="output", batch_size=16):
    """
    Generate handwriting from a saved model
    Args:
        input_text: Text to convert to handwriting
        font_path: Path to the font file
        model_path: Path to the saved generator model
        output_dir: Directory to save the output
        batch_size: Batch size for processing
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()

    # Prepare transform
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Create font
    font = ImageFont.truetype(font_path, 64)

    # Generate individual characters
    characters = []
    char_tensors = []

    for char in input_text:
        if char.isspace():
            characters.append(char)
            continue

        # Create font image
        img = Image.new('L', (128, 128), 255)
        draw = ImageDraw.Draw(img)

        # Get text size
        left, top, right, bottom = font.getbbox(char)
        w, h = right - left, bottom - top

        # Draw text centered
        draw.text(((128 - w) // 2, (128 - h) // 2), char, font=font, fill=0)

        # Transform
        img_tensor = transform(img).unsqueeze(0).to(device)
        char_tensors.append(img_tensor)
        characters.append(char)

        # Process in batches
        if len(char_tensors) == batch_size or (char == input_text[-1] and char_tensors):
            with torch.no_grad():
                # Stack tensors
                batch_tensor = torch.cat(char_tensors, dim=0)

                # Generate handwriting
                fake_handwriting = generator(batch_tensor)

                # Convert to numpy and denormalize
                fake_handwriting = (fake_handwriting * 0.5 + 0.5).cpu().numpy()

                # Clean up background noise
                for i in range(fake_handwriting.shape[0]):
                    fake_handwriting[i, 0] = clean_background(fake_handwriting[i, 0] * 255) / 255

                # Save images
                for i, char in enumerate(characters[-len(char_tensors):]):
                    if not char.isspace():
                        img = Image.fromarray((fake_handwriting[i, 0] * 255).astype(np.uint8))
                        img.save(os.path.join(output_dir, f"{len(characters) - len(char_tensors) + i}_{char}.png"))

                # Clear tensors
                char_tensors = []

    # Combine characters into a handwritten text
    # First, get dimensions of all character images
    char_images = []
    for i, char in enumerate(characters):
        if char.isspace():
            # Add space
            space_width = 64  # Adjust as needed
            space_img = Image.new('L', (space_width, 128), 255)
            char_images.append(space_img)
        else:
            img_path = os.path.join(output_dir, f"{i}_{char}.png")
            if os.path.exists(img_path):
                img = Image.open(img_path)
                char_images.append(img)

    # Calculate total width
    total_width = sum(img.width for img in char_images)
    max_height = max(img.height for img in char_images)

    # Create new image
    handwritten_text = Image.new('L', (total_width, max_height), 255)

    # Paste characters
    x_offset = 0
    for img in char_images:
        handwritten_text.paste(img, (x_offset, 0))
        x_offset += img.width

    # Save combined image
    handwritten_text.save(os.path.join(output_dir, "handwritten_text.png"))

    return os.path.join(output_dir, "handwritten_text.png")

# Function to evaluate the model
def evaluate_model(model_path, test_dataset, output_dir="evaluation"):
    """
    Evaluate the model on a test dataset
    Args:
        model_path: Path to the saved generator model
        test_dataset: Dataset to evaluate on
        output_dir: Directory to save evaluation results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()

    # Create dataloader
    dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Metrics
    mse_losses = []
    mae_losses = []

    with torch.no_grad():
        for i, (font_imgs, real_imgs) in enumerate(dataloader):
            # Configure input
            real_A = font_imgs.to(device)  # Font rendered image
            real_B = real_imgs.to(device)  # Target "real" handwriting

            # Generate fake handwriting
            fake_B = generator(real_A)

            # Calculate metrics
            mse_loss = nn.MSELoss()(fake_B, real_B).item()
            mae_loss = nn.L1Loss()(fake_B, real_B).item()

            mse_losses.append(mse_loss)
            mae_losses.append(mae_loss)

            # Save sample images
            if i < 5:  # Save first 5 batches
                # Convert to numpy arrays
                real_A = (real_A * 0.5 + 0.5).cpu().numpy()
                real_B = (real_B * 0.5 + 0.5).cpu().numpy()
                fake_B = (fake_B * 0.5 + 0.5).cpu().numpy()

                # Clean up background noise
                for j in range(fake_B.shape[0]):
                    fake_B[j, 0] = clean_background(fake_B[j, 0] * 255) / 255

                # Create figure
                fig, axs = plt.subplots(3, 8, figsize=(20, 7))

                # Plot images
                for j in range(min(8, real_A.shape[0])):
                    # Font image
                    axs[0, j].imshow(real_A[j, 0], cmap='gray')
                    axs[0, j].set_title('Font')
                    axs[0, j].axis('off')

                    # Real handwriting
                    axs[1, j].imshow(real_B[j, 0], cmap='gray')
                    axs[1, j].set_title('Target')
                    axs[1, j].axis('off')

                    # Generated handwriting
                    axs[2, j].imshow(fake_B[j, 0], cmap='gray')
                    axs[2, j].set_title('Generated')
                    axs[2, j].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"eval_samples_{i}.png"))
                plt.close()

    # Calculate average metrics
    avg_mse = sum(mse_losses) / len(mse_losses)
    avg_mae = sum(mae_losses) / len(mae_losses)

    print(f"Evaluation Results:")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")

    # Save metrics
    with open(os.path.join(output_dir, "evaluation_metrics.txt"), "w") as f:
        f.write(f"Average MSE: {avg_mse:.4f}\n")
        f.write(f"Average MAE: {avg_mae:.4f}\n")

    return avg_mse, avg_mae

# Main function to run the entire pipeline


def main():
    # Set paths
    ttf_path = "Myfont-Regular.ttf"  # Replace with your font
    real_samples_dir = "real_samples"  # Directory with real handwriting samples (optional)
    output_dir = "handwriting_gan_output"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Train the model
    print("Training the handwriting GAN...")
    train_handwriting_gan(ttf_path, real_samples_dir, output_dir, num_epochs=100)

    # Generate handwriting from a sample text
    print("Generating handwriting sample...")
    sample_text = "Hello World!"
    model_path = os.path.join(output_dir, "generator_final.pth")

    # Call the functions directly, not as methods of train_handwriting_gan
    handwritten_text_path = generate_handwriting(sample_text, ttf_path, model_path, output_dir)

    print(f"Handwritten text saved to: {handwritten_text_path}")

    # Create test dataset for evaluation
    print("Evaluating model...")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    test_dataset = FontDataset(ttf_path, real_samples_dir, transform, augment=False)

    # Call evaluate_model directly
    evaluate_model(model_path, test_dataset, os.path.join(output_dir, "evaluation"))
if __name__ == "__main__":
    main()

