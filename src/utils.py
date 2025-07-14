import os
import shutil
import random
from src.log import logger

def create_test_split(source_dir, test_dir, split_ratio=0.1):
    """
    Moves a small portion of images from each class folder in source_dir
    to a new test_dir, maintaining folder structure.
    """
    try:
        os.makedirs(test_dir, exist_ok=True)
        logger.info(f"Creating test split in '{test_dir}' from '{source_dir}'...")

        for class_name in os.listdir(source_dir):
            class_path = os.path.join(source_dir, class_name)
            test_class_path = os.path.join(test_dir, class_name)

            if not os.path.isdir(class_path):
                continue

            os.makedirs(test_class_path, exist_ok=True)
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
            test_sample_size = max(1, int(len(images) * split_ratio))

            test_images = random.sample(images, test_sample_size)

            for img in test_images:
                src_img = os.path.join(class_path, img)
                dst_img = os.path.join(test_class_path, img)
                shutil.copy2(src_img, dst_img)  # or move() if you want to remove from training set

            logger.info(f"Moved {test_sample_size} images to {test_class_path}")

        logger.info("Test split creation complete.")

    except Exception as e:
        logger.error(f"Error while creating test split: {e}")
        raise
