# src/pipeline.py

from PIL import Image
from src.preprocess_ip_image import preprocess_input_image
from src.model_loader import load_model
from src.inference_utils import predict_class
from src.inference_utils import get_top_k_predictions
from src.log import logger
from src.config import labels

class PlantDiseasePipeline:
    def __init__(self):
        try:
            logger.info("Initializing the prediction pipeline.")
            self.model = load_model()
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error("Error loading model in pipeline.", exc_info=True)
            raise e

    def run(self, image: Image.Image):
        try:
            logger.info("Preprocessing input image(resizing)...")
            processed_image = preprocess_input_image(image)

            logger.info("Running model prediction...")
            prediction = self.model.predict(processed_image)
            class_index = predict_class(prediction)
            label = labels[class_index]
            confidence = float(prediction[0][class_index])

            top_5 = get_top_k_predictions(prediction[0], labels, k=5)

            logger.info(f"Prediction successful: {label} ({confidence:.2f})")

            return {
                "label": label,
                "confidence": confidence,
                "top_5": top_5
            }

        except Exception as e:
            logger.error("Error during prediction in pipeline.", exc_info=True)
            raise e
