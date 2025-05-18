import logging
import sys

def setup_logging(log_level: int = logging.INFO) -> None:
    """Set up basic logging configuration for the project.
    
    Args:
        log_level: Logging level (default: INFO)
    """
    # Create logger
    logger = logging.getLogger("diabetes_predictor")
    logger.setLevel(log_level)
    
    # Create formatter with a simple format
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name.
    
    Args:
        name: Name for the logger (typically __name__ of the module)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"diabetes_predictor.{name}") 