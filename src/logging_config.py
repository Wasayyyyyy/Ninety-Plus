"""
Shared logging configuration for the Premier League ML System.
"""

import logging
import sys

_CONFIGURED = False

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger with standard formatting across all modules.
    """
    global _CONFIGURED
    logger = logging.getLogger(name)
    
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.handlers = [handler]
        _CONFIGURED = True
        
    return logger
