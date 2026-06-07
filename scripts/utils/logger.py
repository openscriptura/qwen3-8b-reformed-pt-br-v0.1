import logging
import os
import sys
from datetime import datetime
from pathlib import Path


def get_logger(script_name: str, level: str | None = None) -> logging.Logger:
    """Return a logger that writes to both stdout and logs/YYYYMMDD_<name>.log.

    Raw API responses should be written separately to logs/raw/ by the caller
    (see VALIDATION_REPORT.md R5).
    """
    log_level_str = level or os.getenv("LOG_LEVEL", "INFO")
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / f"{datetime.now().strftime('%Y%m%d')}_{script_name}.log"

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s  %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger
