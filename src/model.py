from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from src.log import logger

def build_model(input_shape=(128, 128, 3), num_classes=39, learning_rate=0.001):
    """
    Builds and compiles a CNN model for multi-class image classification.

    Args:
        input_shape (tuple): Shape of the input image (H, W, Channels)
        num_classes (int): Number of output classes
        learning_rate (float): Learning rate for the optimizer

    Returns:
        Compiled Keras model
    """
    try:
        logger.info("Building CNN model...")

        model = Sequential()

        # Convolutional Block 1
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Convolutional Block 2
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        #  Convolutional Block 3
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        #  Fully Connected Layer
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', AUC(name='auc')]
        )

        logger.info("Model built and compiled successfully.")
        return model

    except Exception as e:
        logger.error(f"Error while building model: {str(e)}")
        raise
