# utils/logger.py
import logging
import os

_logger = None

def get_logger(log_dir=r'results\logs', name="mcmc"):
    """
    Get a global logger instance.
    If log_dir is provided, a file handler is attached.
    """
    global _logger
    if _logger is not None:
        return _logger

    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    # file handler (only if log_dir provided)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        logger.addHandler(fh)

    _logger = logger
    return _logger
