import os
import logging
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.log import logger

def create_image_generators(data_dir, img_size=(128, 128), batch_size=32, val_split=0.2):
    """
    Creates train and validation image generators using ImageDataGenerator.
    
    Args:
        data_dir (str): Root folder path where images are stored in subfolders per class.
        img_size (tuple): Target image size to resize all images (width, height).
        batch_size (int): Number of images per batch.
        val_split (float): Fraction of data to be used for validation (0.0 to 1.0)

    Returns:
        train_generator, val_generator: The image generators for training and validation.
    """
    try:
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory '{data_dir}' not found.")

        logger.info(f"Creating ImageDataGenerators from directory: {data_dir}")
        logger.info(f"Target image size: {img_size}, Batch size: {batch_size}, Validation split: {val_split}")

        #  Preprocess Type: Rescale pixel values (0–255 to 0–1)
        #  Preprocess Type: Split data into training and validation sets
        datagen = ImageDataGenerator(
            rescale=1./255,        # Normalize RGB pixel values
            validation_split=val_split
        )

        #  Preprocess Type: Image resizing + loading training set
        train_generator = datagen.flow_from_directory(
            data_dir,
            target_size=img_size,   # Resize all images
            batch_size=batch_size,
            class_mode='categorical',  # Use categorical labels (softmax)
            subset='training',
            shuffle=True
        )

        #  Preprocess Type: Image resizing + loading validation set
        val_generator = datagen.flow_from_directory(
            data_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )

        logger.info("Train and validation generators created successfully.")
        logger.info(f"Number of training samples: {train_generator.samples}")
        logger.info(f"Number of validation samples: {val_generator.samples}")

        return train_generator, val_generator

    except Exception as e:
        logger.error(f"Error in create_image_generators: {str(e)}")
        raise
