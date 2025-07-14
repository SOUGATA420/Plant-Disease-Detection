#for evaluation report purpose module

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from src.log import logger


def generate_report(model, test_generator, report_dir="reports/"):
    os.makedirs(report_dir, exist_ok=True)

    # Predict
    y_true = test_generator.classes
    y_probs = model.predict(test_generator)
    y_pred = np.argmax(y_probs, axis=1)
    class_names = list(test_generator.class_indices.keys())

    # Save classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(os.path.join(report_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    logger.info("Classification report saved.")

    # Save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "confusion_matrix.png"))
    plt.close()
    logger.info("Confusion matrix saved.")

    # Save predictions
    df = pd.DataFrame({
        "Actual": [class_names[i] for i in y_true],
        "Predicted": [class_names[i] for i in y_pred]
    })
    df.to_csv(os.path.join(report_dir, "test_predictions.csv"), index=False)
    logger.info("Predictions saved.")

    return report, cm
