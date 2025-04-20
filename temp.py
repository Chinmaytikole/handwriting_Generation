import os
import numpy as np
import cv2
from sklearn.decomposition import PCA
import joblib
import json


def predict_single_letter(model_dir, image_path, show_image=False):
    """
    Predict a single letter from an image using trained HMM models.

    Args:
        model_dir: Directory containing the trained HMM models
        image_path: Path to the image to predict
        show_image: Whether to display the image with prediction

    Returns:
        Tuple of (predicted letter, scores dictionary)
    """
    # Verify files exist
    if not os.path.exists(model_dir):
        print(f"Model directory not found: {model_dir}")
        return None, None

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None, None

    # Load configuration
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            n_features = config.get('n_features', 10)
        except:
            print("Error loading config, using default values")
            n_features = 10
    else:
        n_features = 10

    # Load trained models
    models = {}
    model_files = [f for f in os.listdir(model_dir) if f.endswith('_model.pkl')]
    for model_file in model_files:
        letter = model_file.split('_model.pkl')[0]
        model_path = os.path.join(model_dir, model_file)
        try:
            models[letter] = joblib.load(model_path)
        except Exception as e:
            print(f"Error loading model for letter {letter}: {e}")

    if not models:
        print("No models loaded. Make sure the models are saved properly.")
        return None, None

    # Load PCA if used in training
    pca = None
    pca_path = os.path.join(model_dir, "pca.pkl")
    if os.path.exists(pca_path):
        try:
            pca = joblib.load(pca_path)
        except Exception as e:
            print(f"Error loading PCA model: {e}")

    # Load and process the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Could not read image: {image_path}")
        return None, None

    # Preprocess image and extract features
    features = extract_features(image, n_features)

    if len(features) == 0:
        print("Failed to extract features from the image")
        return None, None

    # Apply PCA if available
    if pca is not None:
        features = pca.transform(features)

    # Calculate scores for each model
    scores = {}
    for letter, model in models.items():
        try:
            scores[letter] = model.score(features)
        except Exception as e:
            scores[letter] = float('-inf')

    if not scores:
        print("Failed to calculate scores")
        return None, None

    # Get the best prediction
    pred_letter = max(scores, key=scores.get)

    return pred_letter, scores


def preprocess_image(image):
    """Preprocess the image for feature extraction."""
    if len(image.shape) == 3:  # Color image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Resize to have consistent dimensions
    resized = cv2.resize(gray, (32, 32))

    # Binarize the image using Otsu's thresholding
    _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def extract_features(image, n_features=10, window_width=5, step=2):
    """Extract features from image using sliding window approach."""
    binary = preprocess_image(image)
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
        if len(features) > n_features:
            features = features[:n_features]
        elif len(features) < n_features:
            features.extend([0] * (n_features - len(features)))

        sequences.append(features)

    return np.array(sequences)


def extract_letter_from_prediction(pred):
    """
    Extract the actual letter from a prediction like 'upper_a' or 'lower_a'.
    Returns the normalized letter (correct case) and the case indicator.
    """
    if pred is None:
        return None, None

    parts = pred.split('_')
    if len(parts) >= 2:
        case = parts[0].lower()  # 'upper' or 'lower'
        letter = parts[1]  # The letter itself (which may be lowercase in both cases)

        # Ensure correct case based on the prefix
        if case == 'upper':
            normalized_letter = letter.upper()
        elif case == 'lower':
            normalized_letter = letter.lower()
        else:
            normalized_letter = letter

        return normalized_letter, case
    else:
        # If prediction format is unexpected, return as is
        return pred, None


def batch_predict_characters(model_dir, characters_dir):
    """
    Run predictions on all images in the characters directory
    and calculate accuracy.

    Args:
        model_dir: Directory containing the trained HMM models
        characters_dir: Directory containing character images
    """
    if not os.path.exists(characters_dir):
        print(f"Characters directory not found: {characters_dir}")
        return

    # Get all images from the directory
    image_files = [f for f in os.listdir(characters_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"No image files found in {characters_dir}")
        return

    print(f"Found {len(image_files)} images in {characters_dir}")
    print("Starting batch prediction...")

    correct_predictions = 0
    total_predictions = 0

    print("\nPrediction Results:")
    print("-" * 80)
    print(f"{'Image':<15} {'Actual':<10} {'Raw Prediction':<20} {'Normalized':<15} {'Match':<10}")
    print("-" * 80)

    for image_file in sorted(image_files,
                             key=lambda x: int(x.split('_')[0]) if x.split('_')[0].isdigit() else float('inf')):
        image_path = os.path.join(characters_dir, image_file)

        # Extract actual letter from filename (after underscore)
        parts = image_file.split('_')
        if len(parts) >= 2:
            actual_letter = parts[1].split('.')[0]  # Remove file extension
        else:
            actual_letter = "Unknown"

        # Make prediction
        raw_prediction, _ = predict_single_letter(model_dir, image_path, show_image=False)

        if raw_prediction is not None:
            # Parse the prediction (e.g., "upper_a" or "lower_a")
            normalized_letter, _ = extract_letter_from_prediction(raw_prediction)

            if normalized_letter is not None:
                # Check if the prediction matches (exact match including case)
                is_match = normalized_letter == actual_letter

                if is_match:
                    correct_predictions += 1
                    match_status = "✓"
                else:
                    match_status = "✗"

                total_predictions += 1

                print(
                    f"{image_file:<15} {actual_letter:<10} {raw_prediction:<20} {normalized_letter:<15} {match_status:<10}")

    # Calculate and display accuracy
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print("\nPrediction Summary:")
        print("-" * 80)
        print(f"Total images processed: {total_predictions}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("\nNo valid predictions were made.")


# Main execution
if __name__ == "__main__":
    import sys

    # Default values
    model_dir = "hmm_models"
    characters_dir = os.path.join("prepare_dataset", "characters")

    # Allow command line arguments to override defaults
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    if len(sys.argv) > 2:
        characters_dir = sys.argv[2]

    # Run batch prediction
    batch_predict_characters(model_dir, characters_dir)