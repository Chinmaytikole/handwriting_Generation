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
                if len(label) == 1 and f"{label}" in file_name:
                    char = label
                    break
                # For multi-character labels like 'col'
                elif len(label) > 1 and f"{label}" in file_name:
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
                    # Check if inversion is needed - improved detection
                    # Count white vs black pixels to determine background/foreground
                    white_count = np.sum(img > 200)
                    black_count = np.sum(img < 50)

                    # If it's white text on black background, invert
                    if white_count < black_count:
                        img = cv2.bitwise_not(img)

                    # Normalize and binarize for cleaner input
                    _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

                    # Resize to standard size while preserving aspect ratio
                    orig_h, orig_w = img.shape[:2]
                    target_size = 64

                    # Make the image square with white padding
                    square_size = max(orig_h, orig_w)
                    square_img = np.ones((square_size, square_size), dtype=np.uint8) * 255

                    # Center the character in the square image
                    y_offset = (square_size - orig_h) // 2
                    x_offset = (square_size - orig_w) // 2
                    square_img[y_offset:y_offset + orig_h, x_offset:x_offset + orig_w] = img

                    # Resize the square image
                    img = cv2.resize(square_img, (target_size, target_size), interpolation=cv2.INTER_AREA)

                    # Make sure the image has proper contrast - text should be black (0) and background white (255)
                    if np.mean(img) > 127:
                        img = cv2.bitwise_not(img)

                    # Add padding to ensure characters don't touch the edge
                    padded_img = np.ones((target_size + 10, target_size + 10), dtype=np.uint8) * 255
                    padded_img[5:5 + target_size, 5:5 + target_size] = img
                    img = padded_img

                    # Normalize for neural network
                    img = img / 255.0

                    # Convert to PIL image and add alpha channel
                    img_pil = Image.fromarray((img * 255).astype(np.uint8), 'L')
                    img_pil = img_pil.convert("RGBA")
                    datas = img_pil.getdata()

                    new_data = []
                    for item in datas:
                        # Change all white (also shades of whites) pixels to transparent
                        if item[0] > 200 and item[1] > 200 and item[2] > 200:
                            new_data.append((255, 255, 255, 0))
                        else:
                            new_data.append(item)

                    img_pil.putdata(new_data)

                    # Save the image with transparent background
                    char_dir = os.path.join(self.segmented_letters_dir, 'transparent', char)
                    os.makedirs(char_dir, exist_ok=True)
                    img_pil.save(os.path.join(char_dir, file_name))

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
                                # Apply the same preprocessing as in load_segmented_letters
                                _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

                                # Make the image square with white padding
                                orig_h, orig_w = img.shape[:2]
                                square_size = max(orig_h, orig_w)
                                square_img = np.ones((square_size, square_size), dtype=np.uint8) * 255

                                # Center the character in the square image
                                y_offset = (square_size - orig_h) // 2
                                x_offset = (square_size - orig_w) // 2
                                square_img[y_offset:y_offset + orig_h, x_offset:x_offset + orig_w] = img

                                # Resize to a standard size
                                img = cv2.resize(square_img, (64, 64), interpolation=cv2.INTER_AREA)

                                # Ensure proper contrast
                                if np.mean(img) > 127:
                                    img = cv2.bitwise_not(img)

                                # Add padding
                                padded_img = np.ones((74, 74), dtype=np.uint8) * 255
                                padded_img[5:69, 5:69] = img
                                img = padded_img

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
                                    # Apply the same preprocessing as above
                                    _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

                                    # Make the image square with white padding
                                    orig_h, orig_w = img.shape[:2]
                                    square_size = max(orig_h, orig_w)
                                    square_img = np.ones((square_size, square_size), dtype=np.uint8) * 255

                                    # Center the character in the square image
                                    y_offset = (square_size - orig_h) // 2
                                    x_offset = (square_size - orig_w) // 2
                                    square_img[y_offset:y_offset + orig_h, x_offset:x_offset + orig_w] = img

                                    # Resize
                                    img = cv2.resize(square_img, (64, 64), interpolation=cv2.INTER_AREA)

                                    # Ensure proper contrast
                                    if np.mean(img) > 127:
                                        img = cv2.bitwise_not(img)

                                    # Add padding
                                    padded_img = np.ones((74, 74), dtype=np.uint8) * 255
                                    padded_img[5:69, 5:69] = img
                                    img = padded_img

                                    img = img / 255.0
                                    self.char_images[char].append(img)

    def build_variation_model(self):
        """Build a VAE model to generate variations of handwritten characters."""
        # Get input shape from actual data
        input_shape = (74, 74, 1)  # Our standard image size with padding

        # Calculate flattened dimension for decoder
        encoder_conv_layers = 3  # Number of conv layers with stride 2
        conv_dim = input_shape[0] // (2 ** encoder_conv_layers)  # Calculate resulting dimension

        # Encoder
        latent_dim = 64
        encoder_inputs = keras.Input(shape=input_shape)

        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)

        # Store the shape before flattening
        shape_before_flatten = keras.backend.int_shape(x)[1:]
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.3)(x)

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

        # Calculate the dense layer size based on the encoder's output shape
        dense_units = np.prod(shape_before_flatten)

        x = layers.Dense(dense_units, activation="relu")(decoder_inputs)
        x = layers.Reshape(shape_before_flatten)(x)

        x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)

        # Final layer with sigmoid activation for black/white output
        decoder_outputs = layers.Conv2DTranspose(1, 5, activation="sigmoid", padding="same")(x)

        # Create models
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")

        # VAE model
        outputs = decoder(encoder(encoder_inputs)[2])
        vae = keras.Model(encoder_inputs, outputs, name="vae")

        # Calculate reconstruction loss
        reconstruction_loss = keras.losses.binary_crossentropy(
            keras.backend.flatten(encoder_inputs),
            keras.backend.flatten(outputs)
        )
        reconstruction_loss = keras.backend.mean(reconstruction_loss)
        reconstruction_loss *= np.prod(input_shape)  # Scale by input dimensions

        # Calculate KL divergence loss
        kl_loss = 1 + z_log_var - keras.backend.square(z_mean) - keras.backend.exp(z_log_var)
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        # Total loss
        vae_loss = keras.backend.mean(reconstruction_loss + 0.1 * kl_loss)
        vae.add_loss(vae_loss)

        # Compile model
        vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005))

        # Store model components
        self.variation_model = vae
        self.encoder = encoder
        self.decoder = decoder

        return vae

    def prepare_training_data(self):
        """Prepare training data for the variation model."""
        # Collect all character images
        all_images = []
        for char in self.char_labels:
            if char in self.char_images and len(self.char_images[char]) > 0:
                all_images.extend(self.char_images[char])

        # Check if we have enough data
        if len(all_images) < 10:
            print(f"Warning: Only {len(all_images)} samples found. Model may not train effectively.")

        # Convert to numpy array and reshape
        X_train = np.array(all_images).reshape(-1, 74, 74, 1)

        # Ensure all values are between 0 and 1
        X_train = np.clip(X_train, 0.0, 1.0)

        return X_train

    def train_variation_model(self, epochs=100, batch_size=16):  # Reduced batch size
        """Train the variation model on character samples."""
        # Prepare training data
        X_train = self.prepare_training_data()

        if len(X_train) == 0:
            raise ValueError("No training data available")

        # Print shape information for debugging
        print(f"Training data shape: {X_train.shape}")

        # Use a smaller batch size if we have limited data
        if len(X_train) < batch_size:
            batch_size = max(1, len(X_train) // 2)
            print(f"Adjusted batch size to {batch_size} due to limited data")

        # Create a validation split
        if len(X_train) > 5:  # Only split if we have enough data
            train_data, val_data = train_test_split(X_train, test_size=0.2, random_state=42)
        else:
            train_data, val_data = X_train, X_train  # Use same data for both if limited

        # Early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # Model checkpoint to save the best model
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'best_vae_model.h5',
            monitor='val_loss',
            save_best_only=True
        )

        # Reduce learning rate on plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )

        # Train with callbacks
        history = self.variation_model.fit(
            train_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(val_data, None),
            callbacks=[early_stopping, checkpoint, reduce_lr],
            verbose=1
        )

        return history

    def train_variation_model(self, epochs=100, batch_size=32):  # Increased epochs and batch size
        """Train the variation model on character samples."""
        X_train = self.prepare_training_data()

        if len(X_train) == 0:
            raise ValueError("No training data available")

        # Create a validation split
        train_data, val_data = train_test_split(X_train, test_size=0.2, random_state=42)

        # Add early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # Add model checkpoint to save the best model
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'best_vae_model.h5',
            monitor='val_loss',
            save_best_only=True
        )

        # Train with callbacks
        self.variation_model.fit(
            train_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(val_data, None),
            callbacks=[early_stopping, checkpoint]
        )

    def generate_char_variations(self, char, num_variations=10):
        """Generate variations of a handwritten character."""
        if char not in self.char_images or not self.char_images[char]:
            return None

        # Choose a random sample of the character as seed
        seed_img = random.choice(self.char_images[char]).reshape(1, 74, 74, 1)

        # Encode the seed to latent space
        z_mean, z_log_var, _ = self.encoder.predict(seed_img)

        variations = []
        for _ in range(num_variations):
            # Sample points from the latent space with controlled randomness
            z_sample = np.random.normal(z_mean, np.exp(0.3 * z_log_var), size=(1, 64))  # Adjusted latent_dim
            # Decode the point to get a variation
            x_decoded = self.decoder.predict(z_sample)

            # Post-process the generated image for better quality
            gen_img = x_decoded[0].copy()

            # Threshold to make it cleaner
            gen_img = (gen_img > 0.5).astype(np.float32)

            variations.append(gen_img)

        self.char_variations[char] = variations
        return variations

    def prepare_variations(self, variations_per_char=15):
        """Generate variations for all characters."""
        self.char_variations = {}
        for char in self.char_labels:
            if char in self.char_images and len(self.char_images[char]) > 0:
                print(f"Generating variations for '{char}'")
                self.generate_char_variations(char, variations_per_char)

    def add_natural_variation(self, image):
        """Add natural variations to make the handwriting look more realistic."""
        # Make a copy to avoid modifying the original
        image = image.copy()

        # Convert to uint8 for OpenCV operations if it's float
        if image.dtype == np.float32 or image.dtype == np.float64:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.copy()

        # Add slight rotation
        angle = np.random.uniform(-5, 5)
        h, w = image_uint8.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image_uint8, M, (w, h), borderValue=255)

        # Vary line thickness slightly
        kernel_size = random.choice([1, 2, 3])
        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            if random.random() > 0.5:
                # Dilate (thicken)
                processed = cv2.dilate(rotated, kernel, iterations=1)
            else:
                # Erode (thin)
                processed = cv2.erode(rotated, kernel, iterations=1)
        else:
            processed = rotated

        # Add slight perspective transform occasionally
        if random.random() > 0.7:
            h, w = processed.shape[:2]
            src_pts = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
            dst_pts = np.float32([
                [0 + random.randint(0, 15), 0 + random.randint(0, 15)],
                [w - 1 - random.randint(0, 15), 0 + random.randint(0, 15)],
                [0 + random.randint(0, 15), h - 1 - random.randint(0, 15)],
                [w - 1 - random.randint(0, 15), h - 1 - random.randint(0, 15)]
            ])
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            processed = cv2.warpPerspective(processed, M, (w, h), borderValue=255)

        # Add noise
        noise = np.random.normal(0, 5, processed.shape).astype(np.int16)
        processed = np.clip(processed.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Convert back to float if needed
        if image.dtype == np.float32 or image.dtype == np.float64:
            return processed / 255.0
        return processed

    def calculate_character_spacing(self, prev_char, current_char):
        """Calculate appropriate spacing between characters (including kerning)."""
        # Base spacing
        base_spacing = 8  # Increased base spacing

        # More comprehensive kerning pairs
        kerning_pairs = {
            ('T', 'a'): -5, ('T', 'e'): -5, ('T', 'o'): -5,
            ('W', 'a'): -3, ('W', 'e'): -3, ('W', 'o'): -3,
            ('V', 'a'): -3, ('F', 'a'): -2, ('P', 'a'): -2,
            ('r', 'n'): -1, ('r', 'm'): -1,
            ('o', 'o'): -1, ('o', 'e'): -1,
            ('n', 'n'): -1, ('m', 'm'): -1,
            ('A', 'V'): -2, ('A', 'W'): -2, ('A', 'T'): -1,
            ('f', 'f'): -2, ('f', 'i'): -1,
            ('L', 'T'): -1, ('P', 'o'): -1,
            ('v', 'a'): -1, ('w', 'a'): -1
        }

        # Apply kerning if the pair exists
        pair = (prev_char, current_char)
        if pair in kerning_pairs:
            return base_spacing + kerning_pairs[pair]

        # Check for wide characters
        wide_chars = ['m', 'w', 'M', 'W', 'G', 'Q', 'O']
        if prev_char in wide_chars:
            return base_spacing + 3

        # Check for narrow characters
        narrow_chars = ['i', 'l', 'I', '!', "'", '.', ',', 't', 'f']
        if prev_char in narrow_chars:
            return base_spacing - 3

        # Random small variations for natural look
        return base_spacing + random.randint(-1, 1)

    def get_char_width(self, char):
        """Get the width of a character based on its samples."""
        if char == ' ':
            return 25  # Standard space width

        if char in self.char_images and self.char_images[char]:
            img = self.char_images[char][0]
            h, w = img.shape[:2]
            # Calculate width based on standard height
            return int(w * (80 / h)) + random.randint(-3, 3)  # Add slight variation

        return 30  # Default width for unknown characters

    def generate_text_image(self, text, line_height=100, baseline_variation=5, variations=True):
        """Generate an image of handwritten text."""
        if not hasattr(self, 'char_variations') or not self.char_variations:
            print("No character variations found. Generating variations...")
            self.prepare_variations()

        # Basic layout calculations
        text_lines = text.split('\n')

        # Estimate the max width needed
        est_avg_char_width = 45  # Estimated average character width
        max_line_length = max(len(line) for line in text_lines)
        max_spacing = 15

        canvas_width = (est_avg_char_width + max_spacing) * max_line_length
        canvas_height = len(text_lines) * line_height + 100  # Added extra padding

        # Create blank canvas (white background)
        canvas = np.ones((canvas_height, canvas_width), dtype=np.uint8) * 255

        # Place characters on canvas
        y_position = 50  # Start with some top margin

        for line in text_lines:
            x_position = 50  # Start with some left margin
            baseline_offset = random.randint(-baseline_variation, baseline_variation)
            prev_char = None

            # Calculate a slight slant angle for the line
            line_angle = random.uniform(-1, 1)

            for char_idx, char in enumerate(line):
                # Handle space character
                if char == ' ':
                    x_position += random.randint(20, 35)  # Variable space width
                    prev_char = char
                    continue

                if char in self.char_images and len(self.char_images[char]) > 0:
                    # Choose character image (either original or variation)
                    if variations and char in self.char_variations and len(self.char_variations[char]) > 0:
                        # Binary image (0 and 1)
                        char_img = random.choice(self.char_variations[char]).copy()
                        # Convert to uint8 0-255 range
                        char_img = (char_img * 255).astype(np.uint8)
                    else:
                        # Float image (0-1)
                        char_img = random.choice(self.char_images[char]).copy()
                        # Convert to uint8 0-255 range
                        char_img = (char_img * 255).astype(np.uint8)

                    # Add natural variation
                    char_img = self.add_natural_variation(char_img)

                    # Resize character while maintaining aspect ratio
                    h, w = char_img.shape[:2]
                    new_h = int(line_height * 0.8)  # Character height as percentage of line height
                    new_w = int(w * (new_h / h))
                    char_img = cv2.resize(char_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

                    # Apply character-specific baseline offset in addition to line baseline
                    char_baseline = baseline_offset
                    if char in 'gjpqy':  # Characters with descenders
                        char_baseline += random.randint(5, 10)
                    elif char in 'bdfhklt':  # Characters with ascenders
                        char_baseline -= random.randint(3, 7)

                    # Calculate spacing based on previous character
                    spacing = 5
                    if prev_char:
                        spacing = self.calculate_character_spacing(prev_char, char)

                    # Apply line slant - adjust y position based on x position and slant angle
                    additional_y = int(x_position * line_angle / 100)

                    # Place character on canvas with baseline variation
                    y_start = y_position + additional_y + char_baseline

                    # Ensure we don't go outside the canvas
                    if (x_position + new_w < canvas_width and
                            y_start + new_h < canvas_height and
                            y_start >= 0):

                        # Get the character area and ensure it's the right size
                        char_area = np.zeros_like(char_img)
                        char_area = np.ones_like(char_img) * 255

                        # Only copy where the character has pixels
                        mask = (char_img < 200)
                        char_area[mask] = 0  # Black text

                        try:
                            canvas[y_start:y_start + new_h, x_position:x_position + new_w][mask] = 0
                        except:
                            # If there's a shape mismatch, skip this character
                            print(f"Warning: Could not place character '{char}' due to boundary issues")

                    # Update x position
                    x_position += new_w + spacing

                else:
                    # Skip unknown characters
                    x_position += 30

                prev_char = char

            # Move to next line with slight variation
            y_position += line_height + random.randint(-5, 5)

        # Trim canvas if needed
        final_height = min(y_position + 50, canvas_height)
        final_width = min(x_position + 100, canvas_width)
        canvas = canvas[:final_height, :final_width]

        return canvas

    def add_document_effects(self, image):
        """Add effects to make the document look like a real scan/photo."""
        # Convert to 3-channel if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Create a subtle paper texture
        paper_texture = np.ones(image.shape, dtype=np.uint8) * 245  # Light base color

        # Add subtle random variations to texture
        noise = np.random.normal(0, 5, image.shape).astype(np.int16)
        paper_texture = np.clip(paper_texture + noise, 235, 255).astype(np.uint8)

        # Add some very subtle "fibrous" patterns
        for _ in range(5000):
            x = random.randint(0, image.shape[1] - 1)
            y = random.randint(0, image.shape[0] - 1)
            length = random.randint(3, 7)
            angle = random.uniform(0, 2 * np.pi)
            dx = int(length * np.cos(angle))
            dy = int(length * np.sin(angle))
            end_x = min(max(0, x + dx), image.shape[1] - 1)
            end_y = min(max(0, y + dy), image.shape[0] - 1)

            # Draw subtle line
            color_variation = random.randint(-5, 5)
            cv2.line(paper_texture, (x, y), (end_y), (240 + color_variation, 240 + color_variation, 240 + color_variation), 1)

        # Blend the paper texture with the original image
        # Convert image to 3 channels if needed
        if len(image.shape) == 2:
            image_colored = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_colored = image.copy()

        # Use the paper texture for background (where original image is white)
        mask = (image_colored > 240).astype(np.uint8)
        result = np.where(mask, paper_texture, image_colored)

        # Add subtle lighting gradient
        h, w = result.shape[:2]
        gradient = np.zeros((h, w, 3), dtype=np.uint8)

        # Create a radial gradient
        center_x, center_y = w // 2, h // 2
        for y in range(h):
            for x in range(w):
                # Calculate distance from center
                dist = np.sqrt((x - center_x)*2 + (y - center_y)*2)
                # Normalize distance
                max_dist = np.sqrt(center_x*2 + center_y*2)
                normalized_dist = dist / max_dist
                # Create subtle darkening at corners
                factor = 1.0 - 0.1 * normalized_dist
                gradient[y, x] = [int(255 * factor)] * 3

        # Apply gradient
        result = cv2.multiply(result.astype(np.float32)/255, gradient.astype(np.float32)/255) * 255
        result = result.astype(np.uint8)

        # Add very subtle blur
        result = cv2.GaussianBlur(result, (3, 3), 0.5)

        # Add subtle noise
        noise = np.random.normal(0, 3, result.shape).astype(np.int16)
        result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Optional: Add very subtle JPEG compression artifacts
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        _, buffer = cv2.imencode('.jpg', result, encode_param)
        result = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        return result

    def generate_document(self, text, add_effects=True, line_spacing=120):
        """Generate a document with handwritten text."""
        # Generate the raw handwritten text image
        text_image = self.generate_text_image(text, line_height=line_spacing)

        # Add document effects if requested
        if add_effects:
            final_image = self.add_document_effects(text_image)
        else:
            # Convert to color if needed
            if len(text_image.shape) == 2:
                final_image = cv2.cvtColor(text_image, cv2.COLOR_GRAY2BGR)
            else:
                final_image = text_image

        return final_image

    def save_model(self, filename="handwriting_model"):
        """Save the variation model."""
        if self.variation_model is not None:
            self.variation_model.save(f"{filename}.h5")
            print(f"Model saved as {filename}.h5")
        else:
            print("No model to save.")

    def load_model(self, filename="handwriting_model.h5"):
        """Load a previously saved variation model."""
        if os.path.exists(filename):
            self.variation_model = keras.models.load_model(
                filename,
                custom_objects={"sampling": lambda args: args[0] + tf.exp(0.5 * args[1]) * tf.random.normal(tf.shape(args[0]))}
            )

            # Recreate encoder and decoder
            encoder_inputs = self.variation_model.input
            _, _, latent_space = self.variation_model.layers[-2].output
            self.encoder = keras.Model(encoder_inputs, [latent_space])

            latent_dim = latent_space.shape[1]
            decoder_inputs = keras.Input(shape=(latent_dim,))
            # Find decoder layer and recreate it
            decoder_layer = None
            for layer in self.variation_model.layers:
                if layer.name == "decoder":
                    decoder_layer = layer
                    break

            if decoder_layer:
                self.decoder = keras.Model(decoder_inputs, decoder_layer(decoder_inputs))
                print(f"Model loaded successfully from {filename}")
            else:
                print("Warning: Could not recreate decoder model")
        else:
            print(f"File {filename} not found.")

    def visualize_samples(self, num_samples=5):
        """Visualize random samples of each character."""
        # Create figure with subplots
        n_chars = len([char for char in self.char_labels if char in self.char_images and len(self.char_images[char]) > 0])
        if n_chars == 0:
            print("No character samples available to visualize.")
            return

        n_cols = min(10, num_samples)
        n_rows = min(10, n_chars)

        plt.figure(figsize=(n_cols * 2, n_rows * 2))

        row = 0
        for char in self.char_labels:
            if char in self.char_images and len(self.char_images[char]) > 0:
                if row >= n_rows:
                    break

                # Get random samples
                samples = self.char_images[char]
                display_samples = random.sample(samples, min(num_samples, len(samples)))

                for i, sample in enumerate(display_samples):
                    if i >= n_cols:
                        break

                    plt.subplot(n_rows, n_cols, row * n_cols + i + 1)
                    plt.imshow(sample, cmap='gray')
                    plt.title(f"'{char}'")
                    plt.axis('off')

                row += 1

        plt.tight_layout()
        plt.show()

    def visualize_variations(self, chars=None, num_variations=5):
        """Visualize variations of specified characters."""
        if not hasattr(self, 'char_variations') or not self.char_variations:
            print("No variations available. Generate variations first.")
            return

        if chars is None:
            # Display first few characters that have variations
            chars = []
            for char in self.char_labels:
                if char in self.char_variations and len(self.char_variations[char]) > 0:
                    chars.append(char)
                    if len(chars) >= 10:  # Limit to 10 characters
                        break

        n_chars = len(chars)
        if n_chars == 0:
            print("No character variations available to visualize.")
            return

        n_cols = min(10, num_variations)
        n_rows = min(10, n_chars)

        plt.figure(figsize=(n_cols * 2, n_rows * 2))

        for row, char in enumerate(chars[:n_rows]):
            if char in self.char_variations and len(self.char_variations[char]) > 0:
                # Get variations
                variations = self.char_variations[char]
                display_vars = variations[:min(num_variations, len(variations))]

                for i, var in enumerate(display_vars):
                    if i >= n_cols:
                        break

                    plt.subplot(n_rows, n_cols, row * n_cols + i + 1)
                    plt.imshow(var, cmap='gray')
                    plt.title(f"'{char}' var {i+1}")
                    plt.axis('off')

        plt.tight_layout()
        plt.show()

    def demo_text_generation(self, text="Hello world!", line_spacing=100):
        """Generate and display a sample of handwritten text."""
        # Generate the text image
        text_image = self.generate_text_image(text, line_height=line_spacing)

        # Display the image
        plt.figure(figsize=(12, 6))
        plt.imshow(text_image, cmap='gray')
        plt.title("Generated Handwritten Text")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        return text_image

    def demo_document(self, text="Hello world!\nThis is a test.", add_effects=True):
        """Generate and display a document with handwritten text."""
        # Generate the document
        doc_image = self.generate_document(text, add_effects)

        # Display the image
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(doc_image, cv2.COLOR_BGR2RGB))
        plt.title("Generated Document")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        return doc_image

    def export_document(self, text, filename="handwritten_document.jpg", add_effects=True, dpi=300):
        """Generate a document and save it to a file."""
        # Generate the document
        doc_image = self.generate_document(text, add_effects)

        # Save the image
        cv2.imwrite(filename, doc_image)
        print(f"Document saved as {filename}")

        return doc_image

    def generate_multi_author_document(self, texts, authors_per_text=None, add_effects=True):
        """
        Generate a document with different handwriting styles for different sections.

        Args:
            texts: List of text sections
            authors_per_text: List of author IDs (can be None to choose randomly)
            add_effects: Whether to add document effects
        """
        if not hasattr(self, 'char_variations') or not self.char_variations:
            print("No character variations found. Generating variations...")
            self.prepare_variations()

        # Generate a blank document with some padding
        total_height = 100  # Initial padding
        max_width = 1000  # Initial estimate

        # First pass: generate all sections and get dimensions
        section_images = []

        for i, text in enumerate(texts):
            # Generate the section
            section_img = self.generate_text_image(text, line_height=120)
            section_images.append(section_img)

            # Update document dimensions
            total_height += section_img.shape[0] + 50  # Add spacing between sections
            max_width = max(max_width, section_img.shape[1] + 100)  # Ensure sufficient width

        # Create the full document
        full_doc = np.ones((total_height, max_width), dtype=np.uint8) * 255

        # Second pass: place sections on full document
        y_offset = 50
        for section_img in section_images:
            h, w = section_img.shape[:2]
            x_offset = 50  # Left margin

            # Place section on document
            full_doc[y_offset:y_offset+h, x_offset:x_offset+w] = section_img

            # Move to next section
            y_offset += h + 50

        # Add document effects if requested
        if add_effects:
            final_image = self.add_document_effects(full_doc)
        else:
            # Convert to color if needed
            final_image = cv2.cvtColor(full_doc, cv2.COLOR_GRAY2BGR)

        return final_image

# Example usage
if __name__ == "__main__":
    # Create generator
    generator = HandwritingGenerator(segmented_letters_dir="segmented_letters")

    # Load letter images
    try:
        generator.load_segmented_letters()
    except FileNotFoundError:
        print("Segmented letters directory not found. Using alternative method.")
        # Use example mapping if needed
        mapping = {
            'A': 'A', 'B': 'B', 'uppercase': 'C',  # Example mapping
            'a': 'a', 'b': 'b', 'lowercase': 'c',
        }
        generator.extract_characters_manually(directory_mapping=mapping)

    # Build and train model if not already trained
    model_path = "handwriting_model.h5"
    if os.path.exists(model_path):
        print("Loading existing model...")
        generator.load_model(model_path)
    else:
        print("Building and training model...")
        generator.build_variation_model()
        generator.train_variation_model()
        generator.save_model()

    # Generate character variations
    generator.prepare_variations()

    # Generate a sample document
    sample_text = "Dear Friend,\n\nI hope this letter finds you well. I wanted to share with you some exciting news about my recent adventures.\n\nLooking forward to hearing from you soon.\n\nBest regards,\nJohn"
    document = generator.demo_document(sample_text)

    # Save the document
    generator.export_document(sample_text, "handwritten_template.jpg")

    # Visualize character samples and variations
    generator.visualize_samples()
    generator.visualize_variations()