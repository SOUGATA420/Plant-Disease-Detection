# used for predict_class(), top_k predictions, etc.
import numpy as np
from src.log import logger

def predict_class(prediction_array):
    try:
        logger.info("Selecting class with highest probability...")
        return int(np.argmax(prediction_array))
    except Exception as e:
        logger.error(f"Error in predict_class: {str(e)}", exc_info=True)
        raise

def get_top_k_predictions(prediction_array, labels, k=5):
    try:
        logger.info("Fetching top-k predictions...")
        indices = prediction_array.argsort()[-k:][::-1]
        return {labels[i]: float(prediction_array[i]) for i in indices}
    except Exception as e:
        logger.error(f"Error in get_top_k_predictions: {str(e)}", exc_info=True)
        raise
