from rembg import remove
from PIL import Image
import os


class BackgroundRemover:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def process_images(self):
        count = 0
        for filename in os.listdir(self.input_dir):
            input_path = os.path.join(self.input_dir, filename)
            output_path = os.path.join(self.output_dir, filename)

            with Image.open(input_path) as image_file:
                output = remove(image_file)
                output.save(output_path)
                count += 1
                print(f"Processed {count}: {filename}")


if __name__ == "__main__":
    input_dir = 'segmented_letters/input'
    output_dir = 'segmented_letters/output'
    remover = BackgroundRemover(input_dir, output_dir)
    remover.process_images()
