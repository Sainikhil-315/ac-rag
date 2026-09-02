"""
utils/logger.py
Configures structured logging for all pipeline stages.
Each log entry includes: timestamp, stage name, level, message.
This feeds the research analysis / ablation logs.
"""

import logging
import sys
from pathlib import Path

from config.settings import LOG_FILE, LOG_LEVEL


def setup_logger(name: str = "ac_rag") -> logging.Logger:
    """
    Returns a configured logger that writes to both console and log file.
    Call once at startup (main.py). All submodules use logging.getLogger(__name__).
    """
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    log_format = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        logger.addHandler(fh)

    return logger
