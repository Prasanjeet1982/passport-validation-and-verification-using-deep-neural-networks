import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from model_architecture import create_passport_validation_model  # Import the model architecture function

# Directories containing the preprocessed data
train_dir = 'datasets/preprocessed_passport_images/train'
val_dir = 'datasets/preprocessed_passport_images/validation'
test_dir = 'datasets/preprocessed_passport_images/test'

# Define image dimensions and batch size
image_size = (128, 128)
batch_size = 32

# Create data generators for training and validation sets
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Create the model
input_shape = (128, 128, 3)  # Input shape matching preprocessed image size
passport_validation_model = create_passport_validation_model(input_shape)

# Define callbacks (e.g., ModelCheckpoint to save the best model during training)
checkpoint_path = 'models/best_model.h5'
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# Compile the model
passport_validation_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 20
history = passport_validation_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    callbacks=[checkpoint]
)

# Save the final trained model
final_model_path = 'models/final_model.h5'
passport_validation_model.save(final_model_path)
