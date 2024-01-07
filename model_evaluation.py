import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directory containing the preprocessed test data
test_dir = 'datasets/preprocessed_passport_images/test'

# Define image dimensions and batch size
image_size = (128, 128)
batch_size = 32

# Create a data generator for the test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False  # Ensure that the order of the predictions matches the order of the images
)

# Load the trained model
model_path = 'models/final_model.h5'  # Path to the saved final trained model
trained_model = load_model(model_path)

# Evaluate the model on the test set
evaluation = trained_model.evaluate(test_generator, verbose=1)

# Display evaluation metrics (e.g., loss and accuracy)
print("Test Loss:", evaluation[0])
print("Test Accuracy:", evaluation[1])
