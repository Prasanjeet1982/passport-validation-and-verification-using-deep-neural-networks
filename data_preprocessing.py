import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Function to preprocess images (resize and normalize)
def preprocess_images(input_folder, output_folder, image_size):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_list = os.listdir(input_folder)
    for image_name in image_list:
        img_path = os.path.join(input_folder, image_name)
        # Open image using PIL
        img = Image.open(img_path)
        # Resize image
        img = img.resize((image_size, image_size))
        # Normalize pixel values (convert to values between 0 and 1)
        img = np.array(img) / 255.0
        
        # Save preprocessed image
        output_path = os.path.join(output_folder, image_name)
        img = Image.fromarray((img * 255).astype(np.uint8))  # Convert back to PIL image
        img.save(output_path)

# Function to split dataset into train, validation, and test sets
def split_dataset(input_folder, train_size, val_size, test_size):
    image_list = os.listdir(input_folder)
    images_train_val, images_test = train_test_split(image_list, test_size=test_size, random_state=42)
    train_val_ratio = val_size / (train_size + val_size)

    images_train, images_val = train_test_split(images_train_val, test_size=train_val_ratio, random_state=42)

    return images_train, images_val, images_test

# Input folder containing pre-downloaded passport images
input_folder = 'datasets/passport_images'

# Output folder for preprocessed images
output_folder = 'datasets/preprocessed_passport_images'
image_size = 128  # Define the desired image size (e.g., 128x128)

# Preprocess images
preprocess_images(input_folder, output_folder, image_size)

# Split dataset into train, validation, and test sets
train_size = 0.7  # Train set size (70%)
val_size = 0.15  # Validation set size (15%)
test_size = 0.15  # Test set size (15%)

images_train, images_val, images_test = split_dataset(output_folder, train_size, val_size, test_size)

# Print the number of images in each set
print(f"Number of images in train set: {len(images_train)}")
print(f"Number of images in validation set: {len(images_val)}")
print(f"Number of images in test set: {len(images_test)}")
