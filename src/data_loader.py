import os
import zipfile
import shutil
import logging
from kaggle.api.kaggle_api_extended import KaggleApi

# Ensure the logs directory exists before creating the log file
os.makedirs("logs", exist_ok=True)

# Setup logging
logging.basicConfig(filename="logs/app.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def authenticate_kaggle():
    """Ensure the Kaggle API key is set up properly."""
    try:
        os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
        if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
            shutil.copy("kaggle.json", os.path.expanduser("~/.kaggle/kaggle.json"))
        os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)
        logging.info("Kaggle authentication complete.")
    except Exception as e:
        logging.error(f"Kaggle authentication failed: {str(e)}")
        raise

def download_and_extract_dataset(dataset_id: str, download_path: str):
    """Download and extract a Kaggle dataset."""
    try:
        authenticate_kaggle()
        api = KaggleApi()
        api.authenticate()

        os.makedirs(download_path, exist_ok=True)
        logging.info(f"Downloading dataset {dataset_id}...")
        api.dataset_download_files(dataset=dataset_id, path=download_path, unzip=True)
        logging.info(f"Dataset downloaded and extracted to {download_path}")
    
    except Exception as e:
        logging.error(f"Error downloading dataset: {str(e)}")
        raise
