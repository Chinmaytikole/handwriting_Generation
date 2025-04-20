from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import warnings

# Suppress specific warning messages
warnings.filterwarnings("ignore", message="Config of the encoder.*")
warnings.filterwarnings("ignore", message="Config of the decoder.*")
warnings.filterwarnings("ignore", message="Some weights of VisionEncoderDecoderModel.*")

class HandwrittenTextRecognizer:
    def __init__(self):
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        self.model.config.decoder_start_token_id = self.model.config.bos_token_id

        # List of allowed words
        self.allowed_words = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            "cat", "dog", "sun", "run", "big", "the", "and", "sky", "hot", "win",
            "happy", "flower", "banana", "hello", "summer", "pencil", "camera", "window", "letter", "coffee",
            "rocket", "travel", "family", "people", "yellow", "important", "education", "beautiful", "handwriting",
            "electricity", "adventure", "celebration", "generation", "information", "chocolate", "waterfall",
            "friendship", "creativity", "direction", "wonderful"
        ]

    def recognize_text(self, image):
        if not isinstance(image, Image.Image):
            raise ValueError("Input should be a PIL.Image object")
        if image.mode != 'RGB':
            image = image.convert('RGB')

        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=20)
        text = self.processor.decode(outputs[0], skip_special_tokens=True).strip()

        # Exact match
        if text in self.allowed_words:
            return text

        # Partial match: first 3 letters
        if len(text) > 3:
            for word in self.allowed_words:
                if word.lower().startswith(text[:3].lower()):
                    return word

        # Return predicted word if no match
        return text

