import logging
import os
from datetime import datetime

#  Create logs directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

#  Log file name with timestamp 
log_filename = os.path.join(log_dir, "app.log")

# 🛠️ Logging configuration
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,  # or DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'  # append mode
)

#  Export logger object for global use
logger = logging.getLogger()
