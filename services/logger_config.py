import logging
import os
from logging.handlers import RotatingFileHandler
from services.config import LOGGER_NAME, LOG_FILE_PATH  # Import from config

def setup_logging():
    """
    Sets up a professional logging configuration for the RAG system.
    Logs are written to a file and also printed to the console.
    """
    # Create a logger instance using imported constant
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)  # Set the minimum level to log

    # Prevent duplicate handlers from being added on re-runs
    if not logger.handlers:
        # Define the log message format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # 1. File Handler: Writes all logs to a file
        # We use RotatingFileHandler to prevent the log file from getting too large.
        # It keeps 5 backup files, each up to 5MB.
        file_handler = RotatingFileHandler(
            LOG_FILE_PATH,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)  # Log INFO and above to the file

        # 2. Console Handler: Prints logs to the console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.DEBUG)  # Log DEBUG and above to the console

        # Add both handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    logger.info("Logging configured successfully.")

def clear_log_file():
    """
    Clears the content of the log file for fresh testing.
    """
    if os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, 'w') as f:
            f.write('')
        print(f"Log file '{LOG_FILE_PATH}' cleared.")
    else:
        print(f"Log file '{LOG_FILE_PATH}' not found.")