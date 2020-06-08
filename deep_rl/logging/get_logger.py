
from deep_rl.logging.configure_logger import configure_logger
import logging

logging.basicConfig(level=logging.INFO)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    configure_logger(logger, level=logging.DEBUG)
    return logger
