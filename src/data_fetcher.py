import os
import zipfile
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi
from src.log import logger  # 🔁 use shared logger

def authenticate_kaggle():
    """Ensure the Kaggle API key is set up properly."""
    try:
        kaggle_dir = os.path.expanduser("~/.kaggle")
        os.makedirs(kaggle_dir, exist_ok=True)

        if not os.path.exists(os.path.join(kaggle_dir, "kaggle.json")):
            shutil.copy("kaggle.json", os.path.join(kaggle_dir, "kaggle.json"))
            os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)

        logger.info("Kaggle authentication complete.")
    except Exception as e:
        logger.error(f" Kaggle authentication failed: {str(e)}", exc_info=True)
        raise

def download_and_extract_dataset(dataset_id: str, download_path: str):
    """Download and extract a Kaggle dataset."""
    try:
        authenticate_kaggle()
        api = KaggleApi()
        api.authenticate()

        os.makedirs(download_path, exist_ok=True)
        logger.info(f"📥 Downloading dataset: {dataset_id}...")
        api.dataset_download_files(dataset=dataset_id, path=download_path, unzip=True)
        logger.info(f" Dataset downloaded and extracted to: {download_path}")
    
    except Exception as e:
        logger.error(f" Error downloading dataset: {str(e)}", exc_info=True)
        raise
