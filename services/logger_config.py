# services/logger_config.py
import logging
from logging.handlers import RotatingFileHandler
from services.config import settings 

def setup_logging():
    """
    Sets up a professional logging configuration for the RAG system.
    Logs are written to a file and also printed to the console.
    """
    logger = logging.getLogger(settings.LOGGER_NAME)
    
    # Avoid adding duplicate handlers if this function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)  # Set the minimum level to process

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )

    # File Handler: Rotates logs to prevent large files.
    try:
        file_handler = RotatingFileHandler(
            settings.LOG_FILE_PATH,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)  # Log INFO and above to the file
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error setting up file logger: {e}")

    # Console Handler: For immediate feedback during development.
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)  # Log DEBUG and above to the console
    logger.addHandler(console_handler)
    
    logger.info("Logging configured successfully.")