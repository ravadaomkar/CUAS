"""
Shared logging setup so every script writes to both the console and a
persistent file under logs/, instead of ad-hoc print() statements.

Usage:
    from utils.logger import get_logger
    log = get_logger("train")       # -> logs/train.log
    log.info("starting run exp01")
"""
import logging
import os
import sys

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")


def get_logger(name: str, log_dir: str = LOG_DIR, level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger named `name` that writes to logs/<name>.log AND stdout.
    Safe to call repeatedly (won't duplicate handlers).
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger  # already configured

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger
