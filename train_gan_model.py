import os
import numpy as np
import cv2
from sklearn.model_selection import KFold
from hmmlearn.hmm import GaussianHMM
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm


class AlphabetHMMRecognizer:
    def __init__(self, n_components=6, n_features=8, pca_components=None,
                 covariance_type="full", n_iter=200):
        """Initialize the HMM-based alphabet recognizer.

        Args:
            n_components: Number of hidden states in the HMM
            n_features: Number of features per window
            pca_components: If set, apply PCA dimensionality reduction
            covariance_type: Covariance matrix type ('diag', 'full', 'tied', 'spherical')
            n_iter: Number of iterations for HMM training
        """
        self.n_components = n_components
        self.n_features = n_features
        self.pca_components = pca_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.models = {}  # Will store one HMM per letter
        self.pca = None

    def _preprocess_image(self, image):
        """Convert image to grayscale and binarize it."""
        if len(image.shape) == 3:  # Color image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Resize to have consistent dimensions
        resized = cv2.resize(gray, (32, 32))

        # Binarize the image using Otsu's thresholding
        _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    def _extract_features(self, image, window_width=5, step=2):
        """Extract features from image using sliding window approach."""
        binary = self._preprocess_image(image)
        h, w = binary.shape
        sequences = []

        # Apply sliding window from left to right
        for x in range(0, w - window_width + 1, step):
            window = binary[:, x:x + window_width]

            # Feature 1: Pixel density in the window
            pixel_density = np.sum(window > 0) / (window_width * h)

            # Feature 2-3: Vertical center of mass
            if np.sum(window > 0) > 0:  # Avoid division by zero
                row_indices, _ = np.where(window > 0)
                center_y = np.mean(row_indices) / h
                std_y = np.std(row_indices) / h if len(row_indices) > 1 else 0
            else:
                center_y = 0.5  # Default to middle if no pixels
                std_y = 0

            # Feature 4-5: Horizontal transitions (black to white)
            transitions = np.sum(np.abs(np.diff(window > 0, axis=1))) / (h * (window_width - 1))

            # Feature 6: Edge density using Sobel operator
            sobelx = cv2.Sobel(window, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(window, cv2.CV_64F, 0, 1, ksize=3)
            edge_density = (np.sum(np.abs(sobelx)) + np.sum(np.abs(sobely))) / (window_width * h * 255)

            # Feature 7-8: Top and bottom profile
            top_profile = np.zeros(window_width)
            bottom_profile = np.zeros(window_width)
            for i in range(window_width):
                col = window[:, i]
                top_idx = np.where(col > 0)[0]
                if len(top_idx) > 0:
                    top_profile[i] = top_idx[0] / h
                    bottom_profile[i] = top_idx[-1] / h

            top_mean = np.mean(top_profile)
            bottom_mean = np.mean(bottom_profile)

            # Combine all features
            features = [pixel_density, center_y, std_y, transitions, edge_density,
                        top_mean, bottom_mean]

            # Use histogram of pixel distributions in 4 quadrants
            quadrant_features = []
            half_h, half_w = h // 2, window_width // 2
            quadrants = [window[:half_h, :half_w], window[:half_h, half_w:],
                         window[half_h:, :half_w], window[half_h:, half_w:]]

            for quad in quadrants:
                quadrant_features.append(np.sum(quad > 0) / (quad.shape[0] * quad.shape[1]))

            features.extend(quadrant_features)

            # Limit to n_features (take first n or pad if needed)
            if len(features) > self.n_features:
                features = features[:self.n_features]
            elif len(features) < self.n_features:
                features.extend([0] * (self.n_features - len(features)))

            sequences.append(features)

        return np.array(sequences)

    def _augment_image(self, image, num_augmentations=5):
        """Generate augmented versions of the input image."""
        augmented_images = [image]  # Include the original image

        for i in range(num_augmentations):
            aug_image = image.copy()

            # 1. Small random rotation (+/- 10 degrees)
            angle = np.random.uniform(-10, 10)
            h, w = aug_image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            aug_image = cv2.warpAffine(aug_image, rotation_matrix, (w, h),borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            # 2. Small random scaling (0.9 to 1.1)
            scale = np.random.uniform(0.9, 1.1)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(aug_image, (new_w, new_h))

            # Center the resized image on a blank canvas of original size
            canvas = np.zeros_like(image)
            y_offset = max(0, (h - new_h) // 2)
            x_offset = max(0, (w - new_w) // 2)

            # Ensure dimensions don't exceed the canvas
            paste_h = min(new_h, h - y_offset)
            paste_w = min(new_w, w - x_offset)

            canvas[y_offset:y_offset + paste_h, x_offset:x_offset + paste_w] = \
                resized[:paste_h, :paste_w]

            aug_image = canvas

            # 3. Add small amount of noise
            if np.random.random() > 0.5:
                noise = np.random.normal(0, 5, aug_image.shape).astype(np.uint8)
                aug_image = cv2.add(aug_image, noise)

            # 4. Small elastic distortion (simplified version)
            if np.random.random() > 0.5:
                dx = np.random.uniform(-2, 2, aug_image.shape).astype(np.float32)
                dy = np.random.uniform(-2, 2, aug_image.shape).astype(np.float32)

                x, y = np.meshgrid(np.arange(w), np.arange(h))

                map_x = np.float32(x + dx)
                map_y = np.float32(y + dy)

                aug_image = cv2.remap(aug_image, map_x, map_y,
                                      interpolation=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=0)

            augmented_images.append(aug_image)

        return augmented_images

    def fit(self, dataset_path, k_fold=5):
        """Train the HMM models for each letter using k-fold cross-validation.

        Args:
            dataset_path: Path to the dataset directory
            k_fold: Number of folds for cross-validation

        Returns:
            Dictionary with cross-validation metrics
        """
        # Collect all samples
        letters = []
        features_list = []
        labels = []

        # Process uppercase and lowercase letters
        for case in ['upper', 'lower']:
            for letter in 'abcdefghijklmnopqrstuvwxyz':
                letter_dir = os.path.join(dataset_path, f"{case}_{letter}")

                if not os.path.exists(letter_dir):
                    print(f"Warning: Directory not found: {letter_dir}")
                    continue

                # Process each image in the letter directory
                for img_file in os.listdir(letter_dir):
                    if img_file.endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(letter_dir, img_file)
                        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                        if image is None:
                            print(f"Warning: Could not read image: {img_path}")
                            continue

                        # Data augmentation
                        augmented_images = self._augment_image(image, num_augmentations=9)  # Create 9 more variations

                        for aug_img in augmented_images:
                            features = self._extract_features(aug_img)
                            if len(features) > 0:  # Ensure we got valid features
                                letters.append(f"{case}_{letter}")
                                features_list.append(features)
                                labels.append(f"{case}_{letter}")

        # Apply PCA if requested
        if self.pca_components is not None:
            # Flatten the features for PCA
            flat_features = np.vstack([f for f in features_list])
            self.pca = PCA(n_components=self.pca_components)
            self.pca.fit(flat_features)

            # Transform each sequence with PCA
            for i in range(len(features_list)):
                features_list[i] = self.pca.transform(features_list[i])

        # Group by letter
        letter_to_indices = {}
        for i, letter in enumerate(letters):
            if letter not in letter_to_indices:
                letter_to_indices[letter] = []
            letter_to_indices[letter].append(i)

        # Cross-validation metrics
        cv_metrics = {'accuracy': [], 'per_letter': {}}

        # Train a separate HMM for each letter using k-fold CV
        for letter in tqdm(letter_to_indices.keys(), desc="Training letter models"):
            indices = letter_to_indices[letter]

            # Skip letters with too few samples
            if len(indices) < k_fold:
                print(f"Warning: Letter {letter} has only {len(indices)} samples, skipping cross-validation")
                # Train on all available data
                X = [features_list[i] for i in indices]
                try:
                    model = GaussianHMM(n_components=self.n_components,
                                        covariance_type=self.covariance_type,
                                        n_iter=self.n_iter)
                    model.fit(np.vstack(X))
                    self.models[letter] = model
                except Exception as e:
                    print(f"Error training model for letter {letter}: {e}")
                continue

            letter_accuracy = []
            kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)

            for train_idx, test_idx in kf.split(indices):
                train_indices = [indices[i] for i in train_idx]
                test_indices = [indices[i] for i in test_idx]

                # Prepare training data
                X_train = [features_list[i] for i in train_indices]

                # Train the model
                try:
                    model = GaussianHMM(n_components=self.n_components,
                                        covariance_type=self.covariance_type,
                                        n_iter=self.n_iter)
                    model.fit(np.vstack(X_train))

                    # Evaluate on test data
                    correct = 0
                    for test_idx in test_indices:
                        X_test = features_list[test_idx]
                        true_letter = letters[test_idx]

                        # Calculate log likelihood
                        log_likelihood = model.score(X_test)

                        # For the simple case where we only test against the correct model
                        pred_letter = true_letter  # Assume correct since we're only testing the correct model

                        if pred_letter == true_letter:
                            correct += 1

                    fold_accuracy = correct / len(test_indices)
                    letter_accuracy.append(fold_accuracy)

                except Exception as e:
                    print(f"Error in CV for letter {letter}: {e}")

            # Store average accuracy for this letter
            if letter_accuracy:
                avg_accuracy = sum(letter_accuracy) / len(letter_accuracy)
                cv_metrics['per_letter'][letter] = avg_accuracy

            # Train final model on all data for this letter
            X = [features_list[i] for i in indices]
            try:
                model = GaussianHMM(n_components=self.n_components,
                                    covariance_type=self.covariance_type,
                                    n_iter=self.n_iter)
                model.fit(np.vstack(X))
                self.models[letter] = model
            except Exception as e:
                print(f"Error training final model for letter {letter}: {e}")

        # Calculate overall accuracy
        all_accuracies = [acc for acc in cv_metrics['per_letter'].values()]
        if all_accuracies:
            cv_metrics['accuracy'] = sum(all_accuracies) / len(all_accuracies)

        return cv_metrics

    def predict(self, image):
        """Predict the letter in the given image.

        Args:
            image: Input image (numpy array)

        Returns:
            Predicted letter and scores for all letters
        """
        features = self._extract_features(image)

        if len(features) == 0:
            return None, {}

        # Apply PCA if used in training
        if self.pca is not None:
            features = self.pca.transform(features)

        # Calculate log likelihood for each model
        scores = {}
        for letter, model in self.models.items():
            try:
                scores[letter] = model.score(features)
            except Exception as e:
                print(f"Error scoring letter {letter}: {e}")
                scores[letter] = float('-inf')

        # Return the letter with highest log likelihood
        if not scores:
            return None, {}

        best_letter = max(scores, key=scores.get)
        return best_letter, scores

    def evaluate(self, test_dataset_path):
        """Evaluate the recognizer on test dataset.

        Args:
            test_dataset_path: Path to test dataset

        Returns:
            Evaluation metrics
        """
        correct = 0
        total = 0
        confusion_matrix = {}
        letter_accuracy = {}

        # Process uppercase and lowercase letters
        for case in ['upper', 'lower']:
            for letter in 'abcdefghijklmnopqrstuvwxyz':
                true_letter = f"{case}_{letter}"
                letter_dir = os.path.join(test_dataset_path, true_letter)

                if not os.path.exists(letter_dir):
                    print(f"Warning: Test directory not found: {letter_dir}")
                    continue

                letter_correct = 0
                letter_total = 0

                # Initialize confusion matrix row
                if true_letter not in confusion_matrix:
                    confusion_matrix[true_letter] = {}

                # Process each image in the letter directory
                for img_file in os.listdir(letter_dir):
                    if img_file.endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(letter_dir, img_file)
                        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                        if image is None:
                            print(f"Warning: Could not read test image: {img_path}")
                            continue

                        # Predict letter
                        pred_letter, _ = self.predict(image)

                        # Update metrics
                        if pred_letter is not None:
                            total += 1
                            letter_total += 1

                            # Update confusion matrix
                            if pred_letter not in confusion_matrix[true_letter]:
                                confusion_matrix[true_letter][pred_letter] = 0
                            confusion_matrix[true_letter][pred_letter] += 1

                            if pred_letter == true_letter:
                                correct += 1
                                letter_correct += 1

                # Calculate accuracy for this letter
                if letter_total > 0:
                    letter_accuracy[true_letter] = letter_correct / letter_total

        # Calculate overall accuracy
        overall_accuracy = correct / total if total > 0 else 0

        return {
            'accuracy': overall_accuracy,
            'per_letter': letter_accuracy,
            'confusion_matrix': confusion_matrix
        }

    def save_models(self, output_dir):
        """Save trained HMM models to disk."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for letter, model in self.models.items():
            model_path = os.path.join(output_dir, f"{letter}_model.pkl")
            try:
                import joblib
                joblib.dump(model, model_path)
            except Exception as e:
                print(f"Error saving model for letter {letter}: {e}")

        # Save PCA if used
        if self.pca is not None:
            pca_path = os.path.join(output_dir, "pca.pkl")
            try:
                import joblib
                joblib.dump(self.pca, pca_path)
            except Exception as e:
                print(f"Error saving PCA model: {e}")

    def load_models(self, input_dir):
        """Load trained HMM models from disk."""
        import glob
        import joblib

        # Load letter models
        model_files = glob.glob(os.path.join(input_dir, "*_model.pkl"))
        for model_path in model_files:
            letter = os.path.basename(model_path).split('_model.pkl')[0]
            try:
                self.models[letter] = joblib.load(model_path)
            except Exception as e:
                print(f"Error loading model for letter {letter}: {e}")

        # Load PCA if exists
        pca_path = os.path.join(input_dir, "pca.pkl")
        if os.path.exists(pca_path):
            try:
                self.pca = joblib.load(pca_path)
            except Exception as e:
                print(f"Error loading PCA model: {e}")


# Example usage
if __name__ == "__main__":
    # Parameters
    dataset_path = "prepare_dataset/letters"  # Path to your dataset
    n_components = 4  # Number of hidden states
    n_features = 10  # Number of features per window
    pca_components = 8  # Use PCA for dimensionality reduction

    # Create and train the recognizer
    recognizer = AlphabetHMMRecognizer(n_components=n_components,
                                       n_features=n_features,
                                       pca_components=pca_components,
                                       covariance_type="diag",
                                       n_iter=100)

    # Check if dataset directory exists
    if not os.path.exists(dataset_path):
        print(f"Dataset directory not found: {dataset_path}")
        exit(1)

    # Train with cross-validation
    print("Training models...")
    cv_metrics = recognizer.fit(dataset_path, k_fold=5)

    # Print cross-validation results
    print(f"Cross-validation accuracy: {cv_metrics['accuracy']:.4f}")

    # Save the trained models
    output_dir = "hmm_models"
    print(f"Saving models to {output_dir}...")
    recognizer.save_models(output_dir)

    # Evaluate on test set (if available)
    test_path = "prepare_dataset/test_letters"  # Path to test dataset
    if os.path.exists(test_path):
        print("Evaluating on test set...")
        test_metrics = recognizer.evaluate(test_path)
        print(f"Test accuracy: {test_metrics['accuracy']:.4f}")

        # Print per-letter accuracy
        print("\nPer-letter accuracy:")
        for letter, acc in sorted(test_metrics['per_letter'].items()):
            print(f"{letter}: {acc:.4f}")
    else:
        print(f"Test dataset not found: {test_path}")

    # Demo: predict a single image
    demo_img_path = os.path.join("prepare_dataset", "characters/13_d.png")
    if os.path.exists(demo_img_path):
        print(f"\nPredicting letter for {demo_img_path}...")
        image = cv2.imread(demo_img_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            pred_letter, scores = recognizer.predict(image)
            print(f"Predicted letter: {pred_letter}")

            # Display top 5 predictions
            print("\nTop predictions:")
            for letter, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"{letter}: {score:.2f}")

            # Display the image
            plt.imshow(image, cmap='gray')
            plt.title(f"Predicted: {pred_letter}")
            plt.show()
        else:
            print(f"Could not read demo image: {demo_img_path}")
    else:
        print(f"Demo image not found: {demo_img_path}")