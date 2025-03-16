import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import cv2
import os
import random
from PIL import Image
import glob
from sklearn.model_selection import train_test_split
import re


class HandwritingGenerator:
    def __init__(self, segmented_letters_dir='segmented_letters'):
        self.segmented_letters_dir = segmented_letters_dir
        self.images_dir = os.path.join(segmented_letters_dir, 'images')
        self.char_images = {}
        self.char_variations = {}
        self.variation_model = None
        self.char_labels = ['!', '"', "'", ',', '.', 'col', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                            'K', 'L', 'M', 'N', 'O',
                            'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
                            'h', 'i', 'j',
                            'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    def load_segmented_letters(self):
        """Load pre-segmented letter images from the directory."""
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory {self.images_dir} not found")

        # Initialize dictionary for each character
        for char in self.char_labels:
            self.char_images[char] = []

        # List all image files
        image_files = glob.glob(os.path.join(self.images_dir, "*.png"))
        print(f"Found {len(image_files)} image files")

        # Print a few examples to understand the filename format
        if image_files:
            print("Example filenames:")
            for i in range(min(5, len(image_files))):
                print(os.path.basename(image_files[i]))

        # Create a mapping of filename patterns to character labels
        # Try to extract character information from filename
        for img_path in image_files:
            file_name = os.path.basename(img_path)

            # Try different methods to identify the character
            char = None

            # Method 1: Check if the letter is directly in the filename
            for label in self.char_labels:
                # For single characters, check if they appear in the filename
                if len(label) == 1 and f"_{label}_" in file_name:
                    char = label
                    break
                # For multi-character labels like 'col'
                elif len(label) > 1 and f"_{label}_" in file_name:
                    char = label
                    break

            # Method 2: Extract using regex pattern if Method 1 failed
            if char is None:
                # Try to extract a number from the filename (index to char_labels)
                match = re.search(r'letter_(\d+)_', file_name)
                if match:
                    index = int(match.group(1))
                    if 0 <= index < len(self.char_labels):
                        char = self.char_labels[index]

            # If character identified, process the image
            if char is not None:
                # Load and preprocess image
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Check if inversion is needed
                    # Count white vs black pixels to determine background/foreground
                    white_count = np.sum(img > 200)
                    black_count = np.sum(img < 50)

                    # If it's white text on black background, invert
                    if white_count < black_count:
                        img = cv2.bitwise_not(img)

                    # Resize to standard size
                    img = cv2.resize(img, (64, 64))

                    # Normalize
                    img = img / 255.0

                    # Add to character collection
                    self.char_images[char].append(img)
            else:
                print(f"Could not identify character for file: {file_name}")

        # Report statistics
        for char in self.char_labels:
            count = len(self.char_images[char])
            if count > 0:
                print(f"Loaded {count} samples for character '{char}'")

        # Check if we have enough data
        total_samples = sum(len(samples) for samples in self.char_images.values())
        if total_samples == 0:
            print("No character samples were loaded successfully.")
            print("Please provide more information about your filename format.")
        else:
            print(f"Successfully loaded {total_samples} character samples.")

    def extract_characters_manually(self, directory_mapping=None):
        """
        Alternative method to load character images using a direct mapping
        of directories to character labels.

        Args:
            directory_mapping: Dictionary mapping subdirectory names to character labels
        """
        if directory_mapping is None:
            # Assume each subdirectory is named after the character
            for char in self.char_labels:
                char_dir = os.path.join(self.segmented_letters_dir, char)
                if os.path.isdir(char_dir):
                    self.char_images[char] = []
                    for img_file in os.listdir(char_dir):
                        if img_file.endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(char_dir, img_file)
                            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            if img is not None:
                                img = cv2.resize(img, (64, 64))
                                img = img / 255.0
                                self.char_images[char].append(img)
        else:
            # Use the provided mapping
            for dir_name, char in directory_mapping.items():
                if char in self.char_labels:
                    char_dir = os.path.join(self.segmented_letters_dir, dir_name)
                    if os.path.isdir(char_dir):
                        self.char_images[char] = []
                        for img_file in os.listdir(char_dir):
                            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                                img_path = os.path.join(char_dir, img_file)
                                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                                if img is not None:
                                    img = cv2.resize(img, (64, 64))
                                    img = img / 255.0
                                    self.char_images[char].append(img)

    def build_variation_model(self):
        """Build a VAE model to generate variations of handwritten characters."""
        # Encoder
        latent_dim = 32
        encoder_inputs = keras.Input(shape=(64, 64, 1))
        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(16, activation="relu")(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

        # Sampling function
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        z = layers.Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])

        # Decoder
        decoder_inputs = keras.Input(shape=(latent_dim,))
        x = layers.Dense(16 * 16 * 64, activation="relu")(decoder_inputs)
        x = layers.Reshape((16, 16, 64))(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)

        # VAE model
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")

        outputs = decoder(encoder(encoder_inputs)[2])
        vae = keras.Model(encoder_inputs, outputs, name="vae")

        # Add VAE loss
        reconstruction_loss = keras.losses.binary_crossentropy(
            keras.backend.flatten(encoder_inputs),
            keras.backend.flatten(outputs)
        )
        reconstruction_loss *= 64 * 64
        kl_loss = 1 + z_log_var - keras.backend.square(z_mean) - keras.backend.exp(z_log_var)
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)

        vae.compile(optimizer='adam')
        self.variation_model = vae
        self.encoder = encoder
        self.decoder = decoder

        return vae

    def prepare_training_data(self):
        """Prepare training data for the variation model."""
        # Collect all character images
        all_images = []
        for char in self.char_labels:
            if char in self.char_images:
                all_images.extend(self.char_images[char])

        # Convert to numpy array and reshape
        X_train = np.array(all_images).reshape(-1, 64, 64, 1)
        return X_train

    def train_variation_model(self, epochs=50, batch_size=16):
        """Train the variation model on character samples."""
        X_train = self.prepare_training_data()

        if len(X_train) == 0:
            raise ValueError("No training data available")

        self.variation_model.fit(
            X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1
        )

    def generate_char_variations(self, char, num_variations=10):
        """Generate variations of a handwritten character."""
        if char not in self.char_images or not self.char_images[char]:
            return None

        # Choose a random sample of the character as seed
        seed_img = random.choice(self.char_images[char]).reshape(1, 64, 64, 1)

        # Encode the seed to latent space
        z_mean, z_log_var, _ = self.encoder.predict(seed_img)

        variations = []
        for _ in range(num_variations):
            # Sample points from the latent space
            z_sample = np.random.normal(z_mean, np.exp(0.5 * z_log_var), size=(1, 32))
            # Decode the point to get a variation
            x_decoded = self.decoder.predict(z_sample)
            variations.append(x_decoded[0])

        self.char_variations[char] = variations
        return variations

    def prepare_variations(self, variations_per_char=10):
        """Generate variations for all characters."""
        self.char_variations = {}
        for char in self.char_labels:
            if char in self.char_images and len(self.char_images[char]) > 0:
                print(f"Generating variations for '{char}'")
                self.generate_char_variations(char, variations_per_char)

    def add_natural_variation(self, image):
        """Add natural variations to make the handwriting look more realistic."""
        # Add slight rotation
        angle = np.random.uniform(-2, 2)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), borderValue=255)

        # Add slight noise
        noise = np.random.normal(0, 0.03, image.shape).astype(np.float32)
        image = np.clip(image + noise, 0, 1)

        # Vary line thickness slightly
        kernel_size = random.choice([1, 2, 3])
        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            image = cv2.dilate(image, kernel, iterations=1)

        return image

    def calculate_character_spacing(self, prev_char, current_char):
        """Calculate appropriate spacing between characters (including kerning)."""
        # Base spacing
        base_spacing = 5

        # Special cases for certain character pairs (kerning)
        kerning_pairs = {
            ('T', 'a'): -5, ('T', 'e'): -5, ('T', 'o'): -5,
            ('W', 'a'): -3, ('W', 'e'): -3, ('W', 'o'): -3,
            ('V', 'a'): -3, ('F', 'a'): -2, ('P', 'a'): -2,
            ('r', 'n'): -1, ('r', 'm'): -1,
            ('o', 'o'): -1, ('o', 'e'): -1,
            ('n', 'n'): -1, ('m', 'm'): -1,
        }

        # Apply kerning if the pair exists
        pair = (prev_char, current_char)
        if pair in kerning_pairs:
            return base_spacing + kerning_pairs[pair]

        # Check for wide characters
        wide_chars = ['m', 'w', 'M', 'W']
        if prev_char in wide_chars:
            return base_spacing + 2

        # Check for narrow characters
        narrow_chars = ['i', 'l', 'I', '!', "'", '.', ',']
        if prev_char in narrow_chars:
            return base_spacing - 2

        return base_spacing

    def connect_characters(self, char1_img, char2_img, spacing):
        """Create a natural connection between characters when appropriate."""
        # Determine if these characters should be connected
        connectable_pairs = [
            ('a', 'b'), ('a', 'n'), ('a', 'm'), ('e', 'n'), ('e', 'r'),
            ('o', 'n'), ('o', 'r'), ('r', 'e'), ('r', 'o'), ('n', 'n'),
            ('n', 'g'), ('i', 'n'), ('i', 'l'), ('m', 'e'), ('m', 'a')
        ]

        # Check if this is the right type of pair
        if any(cp[0] == char1_img and cp[1] == char2_img for cp in connectable_pairs):
            # Find connection points
            # This is simplified - in practice you'd analyze the strokes
            h1, w1 = char1_img.shape[:2]
            h2, w2 = char2_img.shape[:2]

            # Simple connection
            connection = np.ones((h1, spacing), dtype=np.float32)

            # Return the connected images
            return np.hstack([char1_img, connection, char2_img])
        else:
            # Just return them with spacing
            h1, w1 = char1_img.shape[:2]
            spacing_img = np.ones((h1, spacing), dtype=np.float32)
            return np.hstack([char1_img, spacing_img, char2_img])

    def get_char_width(self, char):
        """Get the width of a character based on its samples."""
        if char == ' ':
            return 30  # Standard space width

        if char in self.char_images and self.char_images[char]:
            img = self.char_images[char][0]
            h, w = img.shape[:2]
            # Calculate width based on standard height
            return int(w * (80 * 0.8 / h))

        return 30  # Default width for unknown characters

    def generate_text_image(self, text, line_height=80, baseline_variation=2, variations=True):
        """Generate an image of handwritten text."""
        if not hasattr(self, 'char_variations') or not self.char_variations:
            print("No character variations found. Generating variations...")
            self.prepare_variations()

        # Basic layout calculations
        text_lines = text.split('\n')

        # Estimate the max width needed
        est_avg_char_width = 40  # Estimated average character width
        max_line_length = max(len(line) for line in text_lines)
        max_spacing = 10

        canvas_width = (est_avg_char_width + max_spacing) * max_line_length
        canvas_height = len(text_lines) * line_height

        # Create blank canvas (white background)
        canvas = np.ones((canvas_height, canvas_width), dtype=np.float32)

        # Place characters on canvas
        y_position = 0

        for line in text_lines:
            x_position = 0
            baseline_offset = random.randint(-baseline_variation, baseline_variation)
            prev_char = None

            for char_idx, char in enumerate(line):
                # Handle space character
                if char == ' ':
                    x_position += 30  # Standard space width
                    prev_char = char
                    continue

                if char in self.char_images and len(self.char_images[char]) > 0:
                    # Choose character image (either original or variation)
                    if variations and char in self.char_variations and len(self.char_variations[char]) > 0:
                        char_img = random.choice(self.char_variations[char]).copy()
                    else:
                        char_img = random.choice(self.char_images[char]).copy()

                    # Add natural variation
                    char_img = self.add_natural_variation(char_img)

                    # Resize character while maintaining aspect ratio
                    h, w = char_img.shape[:2]
                    new_h = int(line_height * 0.8)  # Character height as percentage of line height
                    new_w = int(w * (new_h / h))
                    char_img = cv2.resize(char_img, (new_w, new_h))

                    # Calculate spacing based on previous character
                    spacing = 5
                    if prev_char:
                        spacing = self.calculate_character_spacing(prev_char, char)

                    # Place character on canvas with baseline variation
                    y_start = y_position + (line_height - new_h) // 2 + baseline_offset

                    # Ensure we don't go outside the canvas
                    if x_position + new_w < canvas_width and y_start + new_h < canvas_height:
                        canvas[y_start:y_start + new_h, x_position:x_position + new_w] = char_img

                    # Update x position
                    x_position += new_w + spacing

                else:
                    # Skip unknown characters
                    x_position += 30

                prev_char = char

            # Move to next line with slight variation
            y_position += line_height

        # Trim canvas if needed
        if x_position < canvas_width:
            canvas = canvas[:, :x_position + 50]  # Add some right margin

        return (canvas * 255).astype(np.uint8)

    def add_document_effects(self, image):
        """Add effects to make the document look like a real scan/photo."""
        # Convert to 3-channel if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Add slight paper texture
        paper_texture = np.random.normal(235, 10, image.shape).astype(np.uint8)
        paper_texture = cv2.GaussianBlur(paper_texture, (7, 7), 0)

        # Blend the text with the paper texture
        alpha = 0.1  # Strength of the texture effect
        image = cv2.addWeighted(image, 1 - alpha, paper_texture, alpha, 0)

        # Add subtle shadows
        shadow = np.copy(image)
        shadow = cv2.GaussianBlur(shadow, (11, 11), 0)
        shadow = cv2.addWeighted(shadow, 0.3, np.ones_like(shadow) * 220, 0.7, 0)

        # Create a mask for text areas (dark pixels)
        _, mask = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)

        # Apply shadow only around text
        shadow_mask = cv2.bitwise_not(mask)
        image = np.where(shadow_mask[:, :, np.newaxis] == 255, shadow, image)

        # Add a very slight perspective transformation
        pts1 = np.float32([[0, 0], [image.shape[1], 0], [0, image.shape[0]], [image.shape[1], image.shape[0]]])
        pts2 = np.float32([[0, 0],
                           [image.shape[1] - random.randint(0, 30), random.randint(0, 10)],
                           [random.randint(0, 10), image.shape[0] - random.randint(0, 30)],
                           [image.shape[1] - random.randint(0, 30), image.shape[0] - random.randint(0, 10)]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        image = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))

        return image

    def process_document(self, input_text, output_path):
        """Process a document and generate handwritten version."""
        # Generate basic handwritten text
        handwritten_image = self.generate_text_image(input_text)

        # Add page texture and other effects for realism
        handwritten_image = self.add_document_effects(handwritten_image)

        # Save the result
        cv2.imwrite(output_path, handwritten_image)
        print(f"Handwritten document saved to {output_path}")

        return output_path


# Usage example
def main():
    # Initialize the generator
    generator = HandwritingGenerator(segmented_letters_dir='segmented_letters')

    # First try automatic loading
    try:
        generator.load_segmented_letters()

        # Check if any characters were loaded
        total_samples = sum(len(samples) for samples in generator.char_images.values())
        if total_samples == 0:
            raise ValueError("No character samples were loaded with automatic method")

    except Exception as e:
        print(f"Automatic loading failed: {e}")
        print("Trying manual loading method...")

        # If this fails, we can try a more explicit directory structure
        # This assumes you have directories named by index (0, 1, 2, etc.)
        # corresponding to the char_labels array
        directory_mapping = {}
        for i, char in enumerate(generator.char_labels):
            directory_mapping[str(i)] = char

        # Try loading with the manual mapping
        generator.extract_characters_manually(directory_mapping)

    # Build and train the variation model
    generator.build_variation_model()
    generator.train_variation_model(epochs=30, batch_size=16)

    # Generate variations for all characters
    generator.prepare_variations(variations_per_char=15)

    # Process a document
    sample_text = "Hello world!"
    generator.process_document(sample_text, "handwritten_output.png")

    # Display the result
    output = cv2.imread("handwritten_output.png")
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Generated Handwritten Text")
    plt.show()


if __name__ == "__main__":
    main()