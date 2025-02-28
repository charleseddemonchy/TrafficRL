"""
Logging utilities for the Traffic Light Management with Reinforcement Learning project.
"""

import os
import logging

def setup_logger(name="TrafficRL", level=logging.INFO, log_dir="logs"):
    """
    Set up a logger with file and console output.
    
    Args:
        name: Name of the logger
        level: Logging level
        log_dir: Directory to store log files
    
    Returns:
        Configured logger
    """
    # Create the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Return if handlers are already configured
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)
    
    # Setup file handler if directory exists
    try:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, f"{name.lower()}.log"))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    except Exception as e:
        # Log warning to console
        logger.warning(f"Could not set up file logging: {e}")
    
    logger.info(f"Logger initialized: {name}")
    return logger

def enable_debug_logging(logger):
    """
    Enable debug logging for a logger.
    
    Args:
        logger: Logger to enable debug logging on
    """
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers:
        handler.setLevel(logging.DEBUG)
    logger.debug("Debug logging enabled")