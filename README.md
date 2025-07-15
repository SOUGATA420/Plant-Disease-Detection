# 🌿 Plant Leaf Disease Detection App

A multilingual deep learning-powered web application for detecting plant leaf diseases from images. Built with 🧠 TensorFlow and deployed using 🌐 Streamlit on Hugging Face Spaces.

---

## 🚀 APP Features

- 📸 Upload or capture leaf images
- 🔍 Predict disease with confidence score
- 🏆 View Top-5 class predictions
- 🗂️ Maintains recent prediction history
- 🌐 Multilingual Support: English, বাংলা (Bengali), हिन्दी (Hindi)
- ✅ Tips for best results
- uses CNN model 

---

## 🧠 Model Info

**Model Overview**

This model is a Convolutional Neural Network (CNN) implemented using TensorFlow/Keras for image classification. It's trained to classify plant leaf images into 39 different disease or healthy categories.

CNN is chosen because it mimics the way a human eye understands images, and it can automatically learn disease-specific patterns from plant leaves to make accurate predictions.
**Model Architecture Summary**

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #

=================================================================
conv2d (Conv2D)              (None, 126, 126, 32)      896

max_pooling2d (MaxPooling2D) (None, 63, 63, 32)        0

conv2d_1 (Conv2D)            (None, 61, 61, 64)        18,496

max_pooling2d_1             (None, 30, 30, 64)         0

conv2d_2 (Conv2D)            (None, 28, 28, 128)       73,856

max_pooling2d_2             (None, 14, 14, 128)        0

flatten (Flatten)            (None, 25088)             0

dense (Dense)                (None, 256)               6,422,784

dropout (Dropout)            (None, 256)               0

dense_1 (Dense)              (None, 39)                10,023

=================================================================
Total params: 6,526,055
Trainable params: 6,526,055
Non-trainable params: 0
_________________________________________________________________


**Explanation of Layers**

| Layer          | Description                                                            |
| -------------- | ---------------------------------------------------------------------- |
| `Conv2D`       | Extracts spatial features from input images.                           |
| `MaxPooling2D` | Reduces feature map size and computation, preventing overfitting.      |
| `Flatten`      | Converts 3D feature maps into 1D vectors for fully connected layers.   |
| `Dense`        | Fully connected layers used for learning higher-level representations. |
| `Dropout`      | Regularization technique to avoid overfitting.                         |
| Final `Dense`  | Output layer with 39 units (one for each class), using `softmax`.      |


**Hyperparameters Used**

    Parameter	        Value

    Input image size	(128, 128, 3)

    Optimizer	        Adam

    Loss Function	    Categorical Crossentropy

    Metrics	            Accuracy, AUC

    Epochs	            10

    Batch Size	        32

    Validation Split	0.2

    Callbacks	EarlyStopping, ModelCheckpoint, TensorBoard

- **Input Shape:** 128 *128 RGB image
- **Framework:** TensorFlow / Keras
- **Trained On:** Augmented dataset of plant leaves with 38+ disease classes

**Performance**
| Metric         | Score    |
| -------------- | -------- |
| Train Accuracy | \~94.18% |
| Val Accuracy   | \~94.18% |
| Test Accuracy  | \~98.2%  |
| AUC (val/test) | > 0.99   |

**Model File**
Saved as: models/final_model.h5

Format: Keras HDF5 (.h5) ---.h5 stands for Hierarchical Data Format version 5 — a file format used to store large numerical datasets efficiently.

Compatible with load_model() in both Streamlit and production scripts



---

## 🛠️ Project Structure
PLANTA_DISEASE_DETECTION/
├── app/                               # Streamlit web app for end-user interaction

│   └── app.py                         # Main script for the interactive web UI

├── data/                              # Dataset and all data splits

│   ├── train/                         # Training images (organized by class)

│   ├── val/                           # Validation images (organized by class)

│   ├── test/                          # Test images — unseen during training

├── models/                            # Saved model files

│   └── final_model.h5                 # Final trained CNN model

├── my_env/                            # (Optional) Python virtual environment (should be in .gitignore)

├── Reports/                           # Evaluation results and artifacts

│   ├── classification_report.txt      # Text classification report of test results

│   ├── confusion_matrix.png           # Visualization of confusion matrix

│   └── test_predictions.csv           # CSV with model predictions on test data

├── src/                               # Source code for the project

│   ├── __init__.py                    # Python module initialization

│   ├── config.py                      # Central config for paths and constants

│   ├── data_fetcher.py                # Download data using Kaggle API

│   ├── data_loader.py                 # Load data using ImageDataGenerator 

│   ├── inference_utils.py             # Helper functions for model inference like top 5 prediction and confidence matrix

│   ├── log.py                         # Custom logger to record steps/output

│   ├── model.py                       # Defines CNN architecture

│   ├── model_loader.py                # Loads saved models for inference/testing

│   ├── pipeline.py                    # Trains the model end-to-end

│   ├── preprocess_ip_image.py        # Preprocessing for a single input image

│   ├── preprocessing.py              # Training data preprocessing and augmentation

│   ├── report.py                      # Generates evaluation reports and plots

│   ├── test_pipeline.py              # Loads test data and evaluates model

│   └── utils.py                       # Common utility functions (directory checks, plotting, etc.)

├── .gitignore                         # Files/folders to be ignored by Git

├── Plant_Disease_Prediction.ipynb     # Jupyter notebook for prototyping and experiments

├── README.md                          # Project overview and instructions

└── requirements.txt                   # List of Python dependencies for the project


## 📦 Installation (For Local Use)
1. **Clone the repo**:


2. **Create virtual environment & install dependencies**:


pip install -r requirements.txt

3. Run the app locally:
streamlit run streamlit_app.py

## 🌍 Deployment


**app link** : https://huggingface.co/spaces/SOUGATA4/plant-disease-detection

## 🖼️ Input Methods Supported
📁 Upload .jpg or .png

📷 Take photo via webcam

🖋 Paste a Base64 image string


#### 🤝 Contributing
Pull requests and feature requests are welcome! Feel free to fork the repo and submit improvements.

