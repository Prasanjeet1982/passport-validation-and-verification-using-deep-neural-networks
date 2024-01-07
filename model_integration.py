import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

class PassportValidator:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.target_size = (128, 128)  # Adjust based on your model's input size

    def preprocess_image(self, image_path):
        img = Image.open(image_path)
        img = img.resize(self.target_size)
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img

    def validate_passport(self, image_path):
        preprocessed_img = self.preprocess_image(image_path)
        prediction = self.model.predict(preprocessed_img)
        return prediction[0][0]  # Assuming it's a binary classification (valid/invalid)

# Example usage:
if __name__ == "__main__":
    model_path = 'models/final_model.h5'  # Path to the saved final trained model
    
    # Initialize the PassportValidator with the trained model
    passport_validator = PassportValidator(model_path)

    # Example usage: validate a passport image
    image_path = 'path/to/passport_image.jpg'  # Replace with the path to the image
    if not os.path.exists(image_path):
        print("File not found. Please enter a valid file path.")
    else:
        validation_result = passport_validator.validate_passport(image_path)
        if validation_result > 0.5:  # Assuming threshold for validation is 0.5
            print("Passport is valid.")
        else:
            print("Passport is invalid.")
