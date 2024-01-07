import numpy as np
from PIL import Image

# Function to preprocess an image
def preprocess_image(image_path, target_size):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to load a model
def load_saved_model(model_path):
    try:
        from tensorflow.keras.models import load_model
        model = load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Other utility functions...

# For example, you can add more functions related to data preprocessing, file handling, etc.
