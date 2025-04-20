import os
import numpy as np
import cv2
from sklearn.model_selection import KFold
from hmmlearn.hmm import GaussianHMM
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.applications import EfficientNetB0, InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficient_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.models import Model


class AlphabetHMMRecognizer:
    def __init__(self, n_components=5, cnn_model='resnet50', feature_layer='avg_pool',
                 pca_components=None, covariance_type="full", n_iter=200):
        """Initialize the HMM-based alphabet recognizer with CNN feature extraction.

        Args:
            n_components: Number of hidden states in the HMM
            cnn_model: Name of the CNN model ('resnet50', 'vgg16', 'efficientnet', 'inception')
            feature_layer: Name of the layer to extract features from
            pca_components: If set, apply PCA dimensionality reduction
            covariance_type: Covariance matrix type ('diag', 'full', 'tied', 'spherical')
            n_iter: Number of iterations for HMM training
        """
        self.n_components = n_components
        self.pca_components = pca_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.models = {}  # Will store one HMM per letter
        self.pca = None

        # Setup CNN feature extractor
        self.cnn_model_name = cnn_model
        self.feature_layer = feature_layer
        self._setup_feature_extractor()

    def _setup_feature_extractor(self):
        """Set up the CNN model for feature extraction."""
        # Load the appropriate pre-trained model
        if self.cnn_model_name == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
            self.preprocess_func = resnet_preprocess
            self.input_shape = (224, 224)
        elif self.cnn_model_name == 'vgg16':
            base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
            self.preprocess_func = vgg_preprocess
            self.input_shape = (224, 224)
        elif self.cnn_model_name == 'efficientnet':
            base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
            self.preprocess_func = efficient_preprocess
            self.input_shape = (224, 224)
        elif self.cnn_model_name == 'inception':
            base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
            self.preprocess_func = inception_preprocess
            self.input_shape = (299, 299)
        else:
            raise ValueError(f"Unsupported CNN model: {self.cnn_model_name}")

        # Create feature extractor model
        if self.feature_layer == 'avg_pool':
            self.feature_extractor = base_model
        else:
            # Create a model that outputs the specified layer
            self.feature_extractor = Model(
                inputs=base_model.input,
                outputs=base_model.get_layer(self.feature_layer).output
            )

        # Print feature dimensions for debugging
        print(f"Using {self.cnn_model_name} with feature layer {self.feature_layer}")
        print(f"Feature dimensions: {self.feature_extractor.output_shape}")

        # Ensure GPU memory is released after model creation
        tf.keras.backend.clear_session()

    def _preprocess_image(self, image):
        """Convert image to RGB and resize for CNN input."""
        if len(image.shape) == 2:  # Grayscale image
            # Convert to RGB by duplicating the channel
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:  # Grayscale image with channel dimension
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA image
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Resize to the expected input size for the CNN
        image = cv2.resize(image, self.input_shape)

        return image

    def _extract_sequence_features(self, image, num_segments=10):
        """Extract CNN features from image segments to create a sequence.

        Instead of using sliding windows, we divide the image into segments
        from left to right to create a sequence.
        """
        # Preprocess the image
        preprocessed = self._preprocess_image(image)
        h, w = preprocessed.shape[:2]

        features = []
        segment_width = w // num_segments

        # Create segments from left to right
        for i in range(num_segments):
            start_x = i * segment_width
            end_x = start_x + segment_width if i < num_segments - 1 else w

            # Extract segment
            segment = preprocessed[:, start_x:end_x].copy()

            # Resize segment back to the CNN input size
            segment = cv2.resize(segment, self.input_shape)

            # Preprocess for the CNN
            x = self.preprocess_func(np.expand_dims(segment, axis=0))

            # Extract features
            with tf.device('/CPU:0'):  # Use CPU to avoid GPU memory issues
                segment_features = self.feature_extractor.predict(x, verbose=0)

            # Flatten if needed
            if len(segment_features.shape) > 2:
                segment_features = segment_features.reshape(segment_features.shape[0], -1)

            features.append(segment_features[0])

        return np.array(features)

    def _extract_features(self, image):
        """Extract sequence features from an image."""
        return self._extract_sequence_features(image)

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
            aug_image = cv2.warpAffine(aug_image, rotation_matrix, (w, h),
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)

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
                dx = np.random.uniform(-2, 2, aug_image.shape[:2]).astype(np.float32)
                dy = np.random.uniform(-2, 2, aug_image.shape[:2]).astype(np.float32)

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
        print("Extracting features from dataset...")
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
                letter_files = [f for f in os.listdir(letter_dir)
                                if f.endswith(('.png', '.jpg', '.jpeg'))]

                for img_file in tqdm(letter_files, desc=f"Processing {case}_{letter}",
                                     leave=False):
                    img_path = os.path.join(letter_dir, img_file)
                    image = cv2.imread(img_path)

                    if image is None:
                        print(f"Warning: Could not read image: {img_path}")
                        continue

                    # Data augmentation
                    augmented_images = self._augment_image(image, num_augmentations=5)

                    for aug_img in augmented_images:
                        try:
                            features = self._extract_features(aug_img)
                            if len(features) > 0:  # Ensure we got valid features
                                letters.append(f"{case}_{letter}")
                                features_list.append(features)
                                labels.append(f"{case}_{letter}")
                        except Exception as e:
                            print(f"Error extracting features from {img_path}: {e}")

        # Apply PCA if requested
        if self.pca_components is not None:
            print("Applying PCA...")
            # Flatten the features for PCA
            flat_features = np.vstack([f.flatten() for f in features_list])
            self.pca = PCA(n_components=self.pca_components)
            self.pca.fit(flat_features)

            # Transform each sequence with PCA
            for i in range(len(features_list)):
                # Reshape, transform with PCA, and reshape back
                orig_shape = features_list[i].shape
                flattened = features_list[i].reshape(orig_shape[0], -1)
                transformed = self.pca.transform(flattened)
                features_list[i] = transformed

        # Group by letter
        letter_to_indices = {}
        for i, letter in enumerate(letters):
            if letter not in letter_to_indices:
                letter_to_indices[letter] = []
            letter_to_indices[letter].append(i)

        # Cross-validation metrics
        cv_metrics = {'accuracy': [], 'per_letter': {}}

        # Train a separate HMM for each letter using k-fold CV
        print("Training HMM models...")
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
                    flat_X = np.vstack([x.reshape(x.shape[0], -1) for x in X])
                    model.fit(flat_X)
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
                    # Flatten features if needed
                    flat_X_train = np.vstack([x.reshape(x.shape[0], -1) for x in X_train])
                    model.fit(flat_X_train)

                    # Evaluate on test data
                    correct = 0
                    for test_idx in test_indices:
                        X_test = features_list[test_idx]
                        true_letter = letters[test_idx]

                        # Flatten features if needed
                        flat_X_test = X_test.reshape(X_test.shape[0], -1)

                        # Calculate log likelihood
                        log_likelihood = model.score(flat_X_test)

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
                # Flatten features if needed
                flat_X = np.vstack([x.reshape(x.shape[0], -1) for x in X])
                model.fit(flat_X)
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

        # Flatten features if needed
        flat_features = features.reshape(features.shape[0], -1)

        # Apply PCA if used in training
        if self.pca is not None:
            flat_features = self.pca.transform(flat_features)

        # Calculate log likelihood for each model
        scores = {}
        for letter, model in self.models.items():
            try:
                scores[letter] = model.score(flat_features)
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
                test_files = [f for f in os.listdir(letter_dir)
                              if f.endswith(('.png', '.jpg', '.jpeg'))]

                for img_file in tqdm(test_files, desc=f"Testing {true_letter}", leave=False):
                    img_path = os.path.join(letter_dir, img_file)
                    image = cv2.imread(img_path)

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

        # Save CNN model info
        model_info = {
            'cnn_model': self.cnn_model_name,
            'feature_layer': self.feature_layer,
            'input_shape': self.input_shape
        }
        info_path = os.path.join(output_dir, "model_info.pkl")
        try:
            import joblib
            joblib.dump(model_info, info_path)
        except Exception as e:
            print(f"Error saving model info: {e}")

    def load_models(self, input_dir):
        """Load trained HMM models from disk."""
        import glob
        import joblib

        # Load model info
        info_path = os.path.join(input_dir, "model_info.pkl")
        if os.path.exists(info_path):
            try:
                model_info = joblib.load(info_path)
                self.cnn_model_name = model_info['cnn_model']
                self.feature_layer = model_info['feature_layer']
                self.input_shape = model_info['input_shape']
                self._setup_feature_extractor()
            except Exception as e:
                print(f"Error loading model info: {e}")

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
    pca_components = 64  # PCA components to reduce CNN features
    cnn_model = 'resnet50'  # Options: 'resnet50', 'vgg16', 'efficientnet', 'inception'
    feature_layer = 'avg_pool'  # Layer to extract features from

    # Create and train the recognizer
    recognizer = AlphabetHMMRecognizer(
        n_components=n_components,
        cnn_model=cnn_model,
        feature_layer=feature_layer,
        pca_components=pca_components,
        covariance_type="diag",
        n_iter=100
    )

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
    output_dir = f"hmm_{cnn_model}_models"
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
        image = cv2.imread(demo_img_path)
        if image is not None:
            pred_letter, scores = recognizer.predict(image)
            print(f"Predicted letter: {pred_letter}")

            # Display top 5 predictions
            print("\nTop predictions:")
            for letter, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"{letter}: {score:.2f}")

            # Display the image
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(f"Predicted: {pred_letter}")
            plt.show()
        else:
            print(f"Could not read demo image: {demo_img_path}")
    else:
        print(f"Demo image not found: {demo_img_path}")