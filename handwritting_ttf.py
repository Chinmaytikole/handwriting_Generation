import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from torchvision import transforms
from scipy import ndimage
import glob
import re


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


# New dataset class that loads from image directory
class ImageDataset(Dataset):
    def __init__(self, images_dir, transform=None, augment=True):
        """
        Dataset class that loads images from a directory structure
        Args:
            images_dir: Directory containing images in format 'letter_0_X_.png'
            transform: Image transformations
            augment: Whether to apply data augmentation
        """
        self.transform = transform
        self.augment = augment

        # Find all image files
        self.image_files = glob.glob(os.path.join(images_dir, "*.png"))

        # Extract character information from filenames
        # Filename format: letter_0_X_.png where X is the character
        self.char_images = {}

        for img_path in self.image_files:
            # Extract character from filename
            filename = os.path.basename(img_path)
            match = re.search(r'letter_\d+_(.+)_\.png', filename)
            if match:
                char = match.group(1)
                # Store paths for each character
                if char not in self.char_images:
                    self.char_images[char] = []
                self.char_images[char].append(img_path)

        # Create a list of characters and their corresponding images
        self.samples = []
        for char, paths in self.char_images.items():
            for path in paths:
                self.samples.append((char, path))

        print(f"Loaded {len(self.samples)} images for {len(self.char_images)} unique characters")

    def __len__(self):
        return len(self.samples) * (3 if self.augment else 1)  # Multiple augmentations per image

    def apply_augmentation(self, img, aug_type):
        """Apply various augmentations to the image"""
        img_aug = img.copy()

        if aug_type == 1:
            # Slight rotation
            angle = np.random.uniform(-5, 5)
            img_aug = img_aug.rotate(angle, resample=Image.BICUBIC, fillcolor=255)
        elif aug_type == 2:
            # Slight gaussian blur
            img_aug = img_aug.filter(ImageFilter.GaussianBlur(radius=0.5))

        return img_aug

    def __getitem__(self, idx):
        # Determine original sample index and augmentation type
        sample_idx = idx % len(self.samples)
        aug_type = idx // len(self.samples) if self.augment else 0

        # Get character and image path
        char, img_path = self.samples[sample_idx]

        # Load image and convert to grayscale
        img = Image.open(img_path).convert('L')

        # Create a copy for the target image
        # For training pairs, we'll use the same image with slight variation
        img_original = img.copy()

        # Apply augmentation if enabled
        if self.augment and aug_type > 0:
            img = self.apply_augmentation(img, aug_type)

        # Apply transform
        if self.transform:
            img_tensor = self.transform(img)
            img_original_tensor = self.transform(img_original)

        # For now, use the same image as input and target with different augmentations
        # This can be modified if you have paired data (input/target)
        return img_tensor, img_original_tensor


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
def train_handwriting_gan(images_dir, output_dir="output", num_epochs=100):
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
    dataset = ImageDataset(images_dir, transform, augment=True)
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
            real_A = font_imgs.to(device)  # Input image
            real_B = real_imgs.to(device)  # Target image

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
                    # Input image
                    axs[0, i].imshow(test_A[i, 0], cmap='gray')
                    axs[0, i].set_title('Input')
                    axs[0, i].axis('off')

                    # Target image
                    axs[1, i].imshow(test_B[i, 0], cmap='gray')
                    axs[1, i].set_title('Target')
                    axs[1, i].axis('off')

                    # Generated image
                    axs[2, i].imshow(fake_B[i, 0], cmap='gray')
                    axs[2, i].set_title('Generated')
                    axs[2, i].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"samples_epoch_{epoch}.png"))
                plt.close()


# Function to generate handwriting from a saved model
def generate_handwriting(input_text, images_dir, model_path, output_dir="output", batch_size=16):
    """
    Generate handwriting from a saved model
    Args:
        input_text: Text to convert to handwriting
        images_dir: Directory containing character images
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

    # Create a map of characters to image paths
    char_to_img = {}
    image_files = glob.glob(os.path.join(images_dir, "*.png"))

    for img_path in image_files:
        filename = os.path.basename(img_path)
        match = re.search(r'letter_\d+_(.+)_\.png', filename)
        if match:
            char = match.group(1)
            if char not in char_to_img:
                char_to_img[char] = []
            char_to_img[char].append(img_path)

    # Generate individual characters
    characters = []
    char_tensors = []

    for char in input_text:
        if char.isspace():
            characters.append(char)
            continue

        # Skip characters we don't have images for
        if char not in char_to_img or not char_to_img[char]:
            print(f"Warning: No image found for character '{char}', skipping.")
            continue

        # Select a random image for this character
        img_path = np.random.choice(char_to_img[char])

        # Load and transform the image
        img = Image.open(img_path).convert('L')
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
    max_height = max(img.height for img in char_images) if char_images else 128

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
        for i, (input_imgs, target_imgs) in enumerate(dataloader):
            # Configure input
            input_imgs = input_imgs.to(device)
            target_imgs = target_imgs.to(device)

            # Generate fake handwriting
            fake_handwriting = generator(input_imgs)

            # Calculate metrics
            mse_loss = nn.MSELoss()(fake_handwriting, target_imgs).item()
            mae_loss = nn.L1Loss()(fake_handwriting, target_imgs).item()

            mse_losses.append(mse_loss)
            mae_losses.append(mae_loss)

            # Save sample images
            if i < 5:  # Save first 5 batches
                # Convert to numpy arrays
                input_imgs_np = (input_imgs * 0.5 + 0.5).cpu().numpy()
                target_imgs_np = (target_imgs * 0.5 + 0.5).cpu().numpy()
                fake_handwriting_np = (fake_handwriting * 0.5 + 0.5).cpu().numpy()

                # Clean up background noise
                for j in range(fake_handwriting_np.shape[0]):
                    fake_handwriting_np[j, 0] = clean_background(fake_handwriting_np[j, 0] * 255) / 255

                # Create figure
                fig, axs = plt.subplots(3, 8, figsize=(20, 7))

                # Plot images
                for j in range(min(8, input_imgs_np.shape[0])):
                    # Input image
                    axs[0, j].imshow(input_imgs_np[j, 0], cmap='gray')
                    axs[0, j].set_title('Input')
                    axs[0, j].axis('off')

                    # Target image
                    axs[1, j].imshow(target_imgs_np[j, 0], cmap='gray')
                    axs[1, j].set_title('Target')
                    axs[1, j].axis('off')

                    # Generated image
                    axs[2, j].imshow(fake_handwriting_np[j, 0], cmap='gray')
                    axs[2, j].set_title('Generated')
                    axs[2, j].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"eval_samples_{i}.png"))
                plt.close()

    # Calculate average metrics
    avg_mse = sum(mse_losses) / len(mse_losses) if mse_losses else 0
    avg_mae = sum(mae_losses) / len(mae_losses) if mae_losses else 0

    print(f"Evaluation Results:")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")

    # Save metrics
    with open(os.path.join(output_dir, "evaluation_metrics.txt"), "w") as f:
        f.write(f"Average MSE: {avg_mse:.4f}\n")
        f.write(f"Average MAE: {avg_mae:.4f}\n")

    return avg_mse, avg_mae


def main():
    # Set paths
    images_dir = "segmented_letters/output"  # Directory containing character images
    output_dir = "handwriting_gan_output"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Train the model
    print("Training the handwriting GAN...")
    train_handwriting_gan(images_dir, output_dir, num_epochs=100)

    # Generate handwriting from a sample text
    print("Generating handwriting sample...")
    sample_text = "Hello World!"
    model_path = os.path.join(output_dir, "generator_final.pth")

    handwritten_text_path = generate_handwriting(sample_text, images_dir, model_path, output_dir)
    print(f"Handwritten text saved to: {handwritten_text_path}")

    # Create test dataset for evaluation
    print("Evaluating model...")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    test_dataset = ImageDataset(images_dir, transform, augment=False)

    evaluate_model(model_path, test_dataset, os.path.join(output_dir, "evaluation"))


if __name__ == "__main__":
    main()