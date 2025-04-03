import logging
import os
import sys
from datetime import datetime

def setup_logging(log_level=logging.INFO, log_to_file=True):
    """
    Configure logging for the MCC classifier application.
    
    Args:
        log_level (int): The logging level to use (default: logging.INFO).
        log_to_file (bool): Whether to log to a file (default: True).
    """
    # Create logs directory if it doesn't exist
    if log_to_file and not os.path.exists("logs"):
        os.makedirs("logs")
    
    # Create a formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if requested
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(f"logs/mcc_classifier_{timestamp}.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Log initial message
    logger.info("Logging configured for MCC Classifier") 