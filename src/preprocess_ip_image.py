# src/preprocessing/preprocess_image.py
import numpy as np

def preprocess_input_image(image):
    image = image.convert("RGB")
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)
