
import logging
import sys

def configure_logger(
    logger: logging.Logger, 
    level: int, 
    output_format: str='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):

    # sh = logging.StreamHandler(sys.stdout)
    # sh.setLevel(level)
    # sh.setFormatter(logging.Formatter(output_format))
    # logger.addHandler(sh)
    return logger