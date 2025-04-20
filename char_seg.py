import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import imagehash
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import tempfile
import shutil
import string
import fitz  # PyMuPDF
import io
from skimage.feature import hog
from mahotas.features import zernike_moments

# Set page configuration
st.set_page_config(page_title="Handwriting Recognition with HMM", layout="wide")
st.title("Handwriting Recognition with HMM")


# Create directories for saving data
@st.cache_resource
def initialize_dirs():
    temp_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(temp_dir, "characters"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "all_blobs"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "extracted_images"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "prepare_dataset/letters"), exist_ok=True)
    return temp_dir


dataset_path = initialize_dirs()

# Constants for image processing
RESIZE_DIM = (1755, 2531)
MIN_HEIGHT = 35
NEW_HEIGHT = 47
KERNEL = np.ones((17, 7), np.uint8)
ALPHABET = list(string.ascii_uppercase + string.ascii_lowercase)

# Session state to keep track of our application state
if 'extracted_images' not in st.session_state:
    st.session_state.extracted_images = []
if 'labeled_data' not in st.session_state:
    st.session_state.labeled_data = {}
if 'char_models' not in st.session_state:
    st.session_state.char_models = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = "upload"  # States: upload, prepare_dataset, train, recognize


# -------------------- PDF Processing Functions --------------------
def extract_images_from_pdf(pdf_file, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    pdf_bytes = pdf_file.read()

    # Save PDF to a temp file
    temp_pdf_path = os.path.join(output_folder, "temp.pdf")
    with open(temp_pdf_path, "wb") as f:
        f.write(pdf_bytes)

    doc = fitz.open(temp_pdf_path)
    img_paths = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image = Image.open(io.BytesIO(image_bytes))
            image = image.resize(RESIZE_DIM, Image.Resampling.LANCZOS)

            filename = f"image_{page_num + 1}_{img_index + 1}.{image_ext}"
            save_path = os.path.join(output_folder, filename)
            image.save(save_path)
            img_paths.append(save_path)

    return img_paths


def segment_and_save(image_path, base_save_path, letter_list):
    os.makedirs(base_save_path, exist_ok=True)
    start_point = [159, 30]
    end_point = [1700, 129]

    image = cv2.imread(image_path)
    if image is None:
        st.error(f"Failed to load image: {image_path}")
        return 0

    total_saved = 0
    for strip_idx in range(len(letter_list)):
        letter = letter_list[strip_idx]
        prefix = "upper_" if letter.isupper() else "lower_"
        letter_dir = os.path.join(base_save_path, f"{prefix}{letter}")
        os.makedirs(letter_dir, exist_ok=True)

        # Crop the strip
        y1, y2 = start_point[1], end_point[1]
        if y1 >= image.shape[0] or y2 > image.shape[0]:
            break  # Stop if we've gone beyond the image dimensions

        cropped_strip = image[y1:y2, start_point[0]:end_point[0]]

        # Grayscale and threshold
        gray = cv2.cvtColor(cropped_strip, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dilated = cv2.dilate(thresh, KERNEL, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda cnt: cv2.boundingRect(cnt)[0])

        existing_hashes = set()
        char_idx = 1

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h < MIN_HEIGHT:
                continue

            char_img = cropped_strip[y:y + h, x:x + w]
            pil_img = Image.fromarray(cv2.cvtColor(char_img, cv2.COLOR_BGR2RGB))
            hash_val = imagehash.phash(pil_img)
            if hash_val in existing_hashes:
                continue
            existing_hashes.add(hash_val)

            # Resize
            aspect_ratio = w / h
            new_width = int(NEW_HEIGHT * aspect_ratio)
            resized = cv2.resize(char_img, (new_width, NEW_HEIGHT), interpolation=cv2.INTER_AREA)

            # Save
            save_path = os.path.join(letter_dir, f"{char_idx}.png")
            cv2.imwrite(save_path, resized)
            char_idx += 1
            total_saved += 1

        # Move to next strip
        start_point[1] += 93
        end_point[1] += 93

    return total_saved


# -------------------- Advanced Feature Extraction --------------------
def extract_hog_features(image, pixels_per_cell=(8, 8)):
    """Extract HOG (Histogram of Oriented Gradients) features from an image."""
    # Ensure grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize image if not already normalized
    if image.max() > 1.0:
        image = image / 255.0

    features = hog(image, orientations=9, pixels_per_cell=pixels_per_cell,
                   cells_per_block=(2, 2), block_norm='L2-Hys',
                   visualize=False, feature_vector=True)
    return features


def extract_zernike(image, radius=21, degree=8):
    """Extract Zernike moments for shape description."""
    # Ensure grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize and threshold if needed
    if image.max() > 1.0:
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    return zernike_moments(image, radius=radius, degree=degree)


def compute_directions(xy_points):
    """Compute stroke directions from point sequences."""
    deltas = np.diff(xy_points, axis=0)
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])  # dy/dx
    return angles


def compute_velocity(xy_points, timestamps):
    """Compute velocity from point sequences and timestamps."""
    deltas = np.diff(xy_points, axis=0)
    dt = np.diff(timestamps)
    velocities = deltas / dt[:, np.newaxis]
    return velocities


def compute_curvature(xy_points):
    """Compute curvature along a path of points."""
    # Need at least 3 points for curvature
    if len(xy_points) < 3:
        return np.zeros(max(0, len(xy_points) - 2))

    # Get first derivatives
    dx_dt = np.gradient(xy_points[:, 0])
    dy_dt = np.gradient(xy_points[:, 1])

    # Get second derivatives
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)

    # Compute curvature
    curvature = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (dx_dt ** 2 + dy_dt ** 2) ** (3 / 2)

    # Handle potential NaN/Inf values
    curvature = np.nan_to_num(curvature)

    return curvature


def extract_contour_features(image):
    """Extract contour-based features from an image."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Threshold the image
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros(5)  # Return default features if no contours found

    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract features from the contour
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)

    # Calculate circularity
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

    # Calculate aspect ratio of bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = float(w) / h if h > 0 else 0

    # Calculate extent (ratio of contour area to bounding rectangle area)
    extent = float(area) / (w * h) if w * h > 0 else 0

    return np.array([area, perimeter, circularity, aspect_ratio, extent])


def extract_advanced_features_from_image(img):
    """Combine multiple feature extraction methods into one comprehensive feature vector."""
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Resize for consistency
    resized = cv2.resize(gray, (64, 64))

    # Extract HOG features
    hog_features = extract_hog_features(resized)

    # Extract Zernike moments
    zernike_features = extract_zernike(resized)

    # Extract contour features
    contour_features = extract_contour_features(resized)

    # Combine all features
    combined_features = np.hstack([hog_features, zernike_features, contour_features])

    # Return as a 2D array with one row
    return combined_features.reshape(1, -1)


def extract_features_all(images):
    """Extract features from multiple images and return combined feature matrix and lengths."""
    features = []
    lengths = []

    for img in images:
        # Get features for this image
        img_features = extract_advanced_features_from_image(img)
        features.append(img_features)
        lengths.append(len(img_features))

    # Stack all features vertically
    if features:
        return np.vstack(features), lengths
    else:
        return np.array([]), []


# -------------------- Character Extraction --------------------
def extract_characters(image, min_height=35):
    # Convert PIL Image to OpenCV format
    image = np.array(image)
    if len(image.shape) == 3 and image.shape[2] == 4:  # RGBA image
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    elif len(image.shape) == 2:  # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Preprocess image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((17, 7), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # Detect contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda cnt: cv2.boundingRect(cnt)[0])

    # Process and save characters
    char_images = []
    blob_dir = os.path.join(dataset_path, "all_blobs")

    # Load existing hashes
    existing_hashes = set()
    char_dir = os.path.join(dataset_path, "characters")
    for label_dir in os.listdir(char_dir):
        label_path = os.path.join(char_dir, label_dir)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                if img_file.endswith('.png'):
                    img_path = os.path.join(label_path, img_file)
                    existing_img = Image.open(img_path).convert("RGB")
                    hash_val = imagehash.phash(existing_img)
                    existing_hashes.add(hash_val)

    for idx, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        character_img = image[y:y + h, x:x + w]

        # Save raw blob
        blob_filename = f"blob_{idx}.png"
        blob_filepath = os.path.join(blob_dir, blob_filename)
        cv2.imwrite(blob_filepath, character_img)

        # Skip if too short
        if h < min_height:
            continue

        # Check for duplicate using perceptual hash
        pil_img = Image.fromarray(cv2.cvtColor(character_img, cv2.COLOR_BGR2RGB))
        hash_val = imagehash.phash(pil_img)

        if hash_val in existing_hashes:
            continue

        # Resize character to height=47 while maintaining aspect ratio
        new_height = 47  # Explicitly set to 47 as requested
        aspect_ratio = w / h
        new_width = int(new_height * aspect_ratio)
        resized_img = cv2.resize(character_img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        char_images.append(resized_img)
        existing_hashes.add(hash_val)

    return char_images


# -------------------- HMM Training --------------------
def train_char_models(labeled_data):
    char_models = {}
    progress_bar = st.progress(0)
    total = len(labeled_data)

    for i, (label, images) in enumerate(labeled_data.items()):
        if len(images) == 0:
            continue
        X, lengths = extract_features_all(images)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = hmm.GaussianHMM(n_components=4, covariance_type='full', n_iter=500)
        model.fit(X_scaled, lengths)
        char_models[label] = (model, scaler)

        # Update progress
        progress_bar.progress((i + 1) / total)
        st.write(f"Trained char HMM for '{label}' with {len(images)} samples.")

    return char_models


# -------------------- Recognition --------------------
def recognize_character(img, char_models):
    features = extract_advanced_features_from_image(img)
    best_score = float('-inf')
    best_char = None

    for label, (model, scaler) in char_models.items():
        try:
            features_scaled = scaler.transform(features)
            score = model.score(features_scaled)
            if score > best_score:
                best_score = score
                best_char = label
        except:
            continue

    return best_char, best_score


# -------------------- Load labeled data from directory --------------------
def load_labeled_data_from_directory():
    letters_dir = os.path.join(dataset_path, "prepare_dataset/letters")
    labeled_data = {}

    if os.path.exists(letters_dir):
        for letter_dir in os.listdir(letters_dir):
            letter_path = os.path.join(letters_dir, letter_dir)
            if os.path.isdir(letter_path):
                # Extract the label (e.g., "upper_A" -> "A", "lower_a" -> "a")
                if letter_dir.startswith("upper_"):
                    label = letter_dir[6:]  # Remove "upper_"
                elif letter_dir.startswith("lower_"):
                    label = letter_dir[6:]  # Remove "lower_"
                else:
                    continue  # Skip unexpected directory names

                labeled_data[label] = []
                for img_file in os.listdir(letter_path):
                    if img_file.endswith('.png'):
                        img_path = os.path.join(letter_path, img_file)
                        img = cv2.imread(img_path)
                        if img is not None:
                            labeled_data[label].append(img)

    return labeled_data


# -------------------- Streamlit UI Components --------------------

# Step 1: Upload PDF and Prepare Dataset
def upload_and_prepare_dataset():
    st.subheader("Step 1: Upload PDF & Prepare Dataset")

    st.write("""
    Upload a PDF containing template alphabet images. The PDF should have:
    - First page: uppercase letters (A-Z)
    - Second page: lowercase letters (a-z)
    Each row should contain multiple examples of a single letter.
    """)

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Choose PDF with alphabet templates", type=["pdf"])

        if uploaded_file:
            st.write(f"Uploaded: {uploaded_file.name}")

            if st.button("Process PDF and Create Dataset"):
                with st.spinner("Processing PDF and extracting characters..."):
                    extracted_folder = os.path.join(dataset_path, "extracted_images")
                    letters_folder = os.path.join(dataset_path, "prepare_dataset/letters")

                    # Extract images from PDF
                    img_paths = extract_images_from_pdf(uploaded_file, extracted_folder)
                    st.write(f"Extracted {len(img_paths)} images from PDF")

                    # Process each image
                    total_chars = 0
                    for i, img_path in enumerate(img_paths):
                        if i == 0:
                            # First page should be uppercase letters
                            chars = segment_and_save(img_path, letters_folder, list(string.ascii_uppercase))
                            st.write(f"Processed uppercase letters, extracted {chars} characters")
                            total_chars += chars
                        elif i == 1:
                            # Second page should be lowercase letters
                            chars = segment_and_save(img_path, letters_folder, list(string.ascii_lowercase))
                            st.write(f"Processed lowercase letters, extracted {chars} characters")
                            total_chars += chars
                        else:
                            st.warning(f"Unexpected extra image found in PDF (page {i + 1})")

                    if total_chars > 0:
                        st.success(f"Successfully created dataset with {total_chars} character images!")
                        st.session_state.current_step = "train"
                        st.rerun()
                    else:
                        st.error("No characters could be extracted from the PDF. Please check the PDF format.")

    with col2:
        st.write("### Alternative: Upload Individual Images")
        uploaded_files = st.file_uploader("Choose images with characters to extract",
                                          type=["jpg", "jpeg", "png"],
                                          accept_multiple_files=True,
                                          key="individual_images")

        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} images")

            if st.button("Extract Characters from Images"):
                extracted_chars = []

                with st.spinner("Extracting characters from all images..."):
                    for uploaded_file in uploaded_files:
                        image = Image.open(uploaded_file)
                        st.image(image, caption=f'Processing: {uploaded_file.name}', use_column_width=True, width=300)

                        # Extract characters from this image
                        chars = extract_characters(image)
                        extracted_chars.extend(chars)
                        st.write(f"Extracted {len(chars)} characters from {uploaded_file.name}")

                st.session_state.extracted_images = extracted_chars

                if len(st.session_state.extracted_images) > 0:
                    st.success(f"Successfully extracted {len(st.session_state.extracted_images)} characters in total!")
                    st.write("Note: You'll need to label these characters manually in the next step.")
                    if st.button("Proceed to Labeling"):
                        st.session_state.current_step = "label"
                        st.rerun()
                else:
                    st.error(
                        "No characters found in any of the images. Try adjusting the parameters or different images.")


# Step 2: Label Characters (only used for individual images)
def label_characters():
    st.subheader("Step 2: Label Extracted Characters")

    if len(st.session_state.extracted_images) == 0:
        st.warning("No characters to label. Please extract characters first.")
        if st.button("Return to Upload"):
            st.session_state.current_step = "upload"
            st.rerun()
        return

    st.write("Label each character by typing the correct letter/number in the text box below it.")

    # Display characters in a grid with text inputs
    cols = st.columns(5)
    labels = {}

    for i, img in enumerate(st.session_state.extracted_images):
        col_idx = i % 5
        with cols[col_idx]:
            # Convert OpenCV image to PIL for Streamlit
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            st.image(pil_img, caption=f"Character {i + 1}", width=100)
            char_label = st.text_input(f"Label for char {i + 1}", key=f"label_{i}", max_chars=1)
            labels[i] = char_label

    if st.button("Save Labels and Train Model"):
        # Check if all characters have been labeled
        if all(label.strip() != "" for label in labels.values()):
            # Save labeled images in the prepare_dataset structure
            letters_dir = os.path.join(dataset_path, "prepare_dataset/letters")

            for i, img in enumerate(st.session_state.extracted_images):
                if labels[i].strip() != "":
                    label = labels[i]

                    # Determine prefix based on case
                    prefix = "upper_" if label.isupper() else "lower_"
                    label_dir = os.path.join(letters_dir, f"{prefix}{label}")
                    os.makedirs(label_dir, exist_ok=True)

                    # Count existing files to get next index
                    existing_files = [f for f in os.listdir(label_dir) if f.endswith('.png')]
                    next_idx = len(existing_files) + 1

                    # Save image
                    filename = f"{next_idx}.png"
                    filepath = os.path.join(label_dir, filename)
                    cv2.imwrite(filepath, img)

            st.success("Characters labeled and saved successfully!")
            st.session_state.current_step = "train"
            st.rerun()
        else:
            st.error("Please label all characters before proceeding.")

    if st.button("Return to Upload"):
        st.session_state.current_step = "upload"
        st.rerun()


# Step 3: Train HMM Models
def train_models():
    st.subheader("Step 3: Train HMM Models")

    # Load labeled data from prepared dataset
    labeled_data = load_labeled_data_from_directory()

    # Update session state with labeled data
    st.session_state.labeled_data = labeled_data

    if not st.session_state.labeled_data:
        st.warning("No labeled data available. Please prepare dataset first.")
        if st.button("Return to Dataset Preparation"):
            st.session_state.current_step = "upload"
            st.rerun()
        return

    # Show summary of labeled data
    st.write("### Dataset Summary")
    st.write(f"Total unique characters: {len(st.session_state.labeled_data)}")

    # Display character stats with columns for uppercase and lowercase
    col1, col2 = st.columns(2)

    with col1:
        st.write("#### Uppercase Letters")
        uppercase_data = {k: v for k, v in st.session_state.labeled_data.items() if k.isupper()}
        for label, images in uppercase_data.items():
            st.write(f"Character '{label}': {len(images)} samples")

    with col2:
        st.write("#### Lowercase Letters")
        lowercase_data = {k: v for k, v in st.session_state.labeled_data.items() if k.islower()}
        for label, images in lowercase_data.items():
            st.write(f"Character '{label}': {len(images)} samples")

    if st.button("Train HMM Models"):
        with st.spinner("Training character models..."):
            st.session_state.char_models = train_char_models(st.session_state.labeled_data)

        st.success("Training complete!")
        st.session_state.current_step = "recognize"
        st.rerun()

    if st.button("Return to Dataset Preparation"):
        st.session_state.current_step = "upload"
        st.rerun()


# Step 4: Recognition
def perform_recognition():
    st.subheader("Step 4: Character Recognition")

    if not st.session_state.char_models:
        st.warning("No trained models available. Please train models first.")
        if st.button("Return to Training"):
            st.session_state.current_step = "train"
            st.rerun()
        return

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Upload an image for recognition", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Image for Recognition', use_column_width=True)

            if st.button("Recognize Character"):
                with st.spinner("Processing..."):
                    # Convert PIL image to OpenCV format
                    img_cv = np.array(image)
                    if len(img_cv.shape) == 3 and img_cv.shape[2] == 4:  # RGBA
                        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2BGR)
                    elif len(img_cv.shape) == 2:  # Grayscale
                        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)

                    with col2:
                        st.write("### Recognition Results")

                        # Resize to height=47 while maintaining aspect ratio
                        h, w = img_cv.shape[:2]
                        new_height = 47
                        aspect_ratio = w / h
                        new_width = int(new_height * aspect_ratio)
                        resized_img = cv2.resize(img_cv, (new_width, new_height), interpolation=cv2.INTER_AREA)

                        char, score = recognize_character(resized_img, st.session_state.char_models)
                        st.write(f"Recognized Character: **{char}**")
                        st.write(f"Confidence Score: {score:.2f}")

                        # Display feature visualization
                        st.write("### Feature Analysis")

                        # Convert to grayscale for visualization
                        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

                        # Show HOG visualization if needed
                        st.write("Using advanced features:")
                        st.write("- HOG (Histogram of Oriented Gradients)")
                        st.write("- Zernike Moments")
                        st.write("- Contour-based Features")

    # Add option to upload more training data
    if st.button("Add More Training Data"):
        st.session_state.current_step = "upload"
        st.rerun()

    if st.button("Start Over"):
        # Reset only the recognition step, keep the trained model
        st.session_state.current_step = "recognize"
        st.rerun()


# -------------------- Main Application --------------------

# Sidebar
st.sidebar.title("Navigation")
steps = ["Upload & Prepare Dataset", "Train Models", "Recognition"]
step_keys = ["upload", "train", "recognize"]

for i, step in enumerate(steps):
    if st.sidebar.button(step):
        st.session_state.current_step = step_keys[i]
        st.rerun()

st.sidebar.divider()
st.sidebar.write("### Current Progress")
current_step_idx = step_keys.index(st.session_state.current_step) if st.session_state.current_step in step_keys else 0
for i, step in enumerate(steps):
    if i < current_step_idx:
        st.sidebar.success(f"✓ {step}")
    elif i == current_step_idx:
        st.sidebar.info(f"➤ {step}")
    else:
        st.sidebar.write(f"○ {step}")

# Display feature extraction information in sidebar
st.sidebar.divider()
st.sidebar.write("### Feature Extraction")
st.sidebar.write("Using:")
st.sidebar.write("✅ HOG features")
st.sidebar.write("✅ Zernike moments")
st.sidebar.write("✅ Contour features")

# Display dataset statistics in sidebar
st.sidebar.divider()
st.sidebar.write("### Dataset Statistics")
letters_dir = os.path.join(dataset_path, "prepare_dataset/letters")
if os.path.exists(letters_dir):
    upper_count = 0
    lower_count = 0
    letter_counts = {}

    for letter_dir in os.listdir(letters_dir):
        letter_path = os.path.join(letters_dir, letter_dir)
        if os.path.isdir(letter_path):
            img_count = len([f for f in os.listdir(letter_path) if f.endswith('.png')])
            letter_counts[letter_dir] = img_count

            if letter_dir.startswith("upper_"):
                upper_count += 1
            elif letter_dir.startswith("lower_"):
                lower_count += 1

    if letter_counts:
        st.sidebar.write(f"Total character classes: {len(letter_counts)}")
        st.sidebar.write(f"Uppercase letters: {upper_count}")
        st.sidebar.write(f"Lowercase letters: {lower_count}")
        st.sidebar.write(f"Total samples: {sum(letter_counts.values())}")
else:
    st.sidebar.write("No characters in dataset yet")

# Add information about the advanced features being used
st.sidebar.divider()
st.sidebar.write("### Model Details")
st.sidebar.write("HMM with advanced features:")
st.sidebar.write("- HOG captures local edge directionality")
st.sidebar.write("- Zernike moments for rotational invariance")
st.sidebar.write("- Contour analysis for shape recognition")

# Display current step
if st.session_state.current_step == "upload":
    upload_and_prepare_dataset()
elif st.session_state.current_step == "label":
    label_characters()
elif st.session_state.current_step == "train":
    train_models()
elif st.session_state.current_step == "recognize":
    perform_recognition()


# Clean up on app close
def on_shutdown():
    shutil.rmtree(dataset_path, ignore_errors=True)


# Register cleanup function
import atexit

atexit.register(on_shutdown)