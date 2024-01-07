import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Function to preprocess the user-uploaded image
def preprocess_user_image(image_path, target_size):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to perform passport validation using the trained model
def validate_passport(image_path, model):
    target_size = (128, 128)  # Adjust based on your model's input size
    preprocessed_img = preprocess_user_image(image_path, target_size)
    prediction = model.predict(preprocessed_img)
    return prediction[0][0]  # Assuming it's a binary classification (valid/invalid)

# Path to the saved final trained model
model_path = 'models/final_model.h5'  

# Load the trained model
trained_model = load_model(model_path)

# User interaction loop
while True:
    print("Please enter the path to the passport image you want to validate (or 'exit' to quit):")
    user_input = input()
    
    if user_input.lower() == 'exit':
        break
    
    if not os.path.exists(user_input):
        print("File not found. Please enter a valid file path.")
        continue
    
    # Perform passport validation
    validation_result = validate_passport(user_input, trained_model)
    
    # Interpret the validation result
    if validation_result > 0.5:  # Assuming threshold for validation is 0.5
        print("Passport is valid.")
    else:
        print("Passport is invalid.")
