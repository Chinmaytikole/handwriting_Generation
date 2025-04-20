import fitz  # PyMuPDF
from PIL import Image
import os
import io

def extract_and_resize_images(pdf_path, output_folder, width=1755, height=2531):
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    img_count = 0

    for page_index in range(len(doc)):
        page = doc[page_index]
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"image_{page_index+1}_{img_index+1}.{image_ext}"

            # Load image using PIL
            image = Image.open(io.BytesIO(image_bytes))
            image_resized = image.resize((width, height), Image.LANCZOS)

            # Save resized image
            image_resized.save(os.path.join(output_folder, image_filename))
            img_count += 1

    print(f"Extracted and resized {img_count} images to '{output_folder}'")

# Example usage
extract_and_resize_images("All_alphabets.pdf", "extracted_images")
