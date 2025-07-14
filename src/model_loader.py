# src/model_loader.py

import os
import tensorflow as tf
from src.log import logger

def load_model(model_path="models/final_model.h5"):
    """
    Load a trained model for prediction.
    
    Args:
        model_path (str): Path to the saved .h5 model file.

    Returns:
        tf.keras.Model: Loaded model ready for inference.
    """
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")

        logger.info(f"Loading model from {model_path}")
        model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully")

        return model

    except Exception as e:
        logger.error("Failed to load model", exc_info=True)
        raise e
