import logging
import os
from typing import Optional


_LEVEL_NAMES = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


def coerce_level(level_value: object, default: int = logging.INFO) -> int:
    if isinstance(level_value, int):
        return level_value
    if isinstance(level_value, str):
        upper = level_value.strip().upper()
        return _LEVEL_NAMES.get(upper, default)
    return default


def setup_logging(level: object = None, *, name: Optional[str] = None) -> logging.Logger:
    """Initialize root logging once and return a configured logger.

    - Global level comes from argument, env LOG_LEVEL, or defaults to INFO.
    - Format includes time, level, logger name, and message.
    - Idempotent: multiple calls won't add duplicate handlers.
    """
    env_level = os.getenv("LOG_LEVEL")
    effective_level = coerce_level(level if level is not None else env_level)

    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)

    root.setLevel(effective_level)

    logger_name = name if name is not None else __name__
    logger = logging.getLogger(logger_name)
    logger.debug("Logging initialized with level=%s", logging.getLevelName(effective_level))
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a module-specific logger. Ensure setup_logging was called once."""
    return logging.getLogger(name if name is not None else __name__)


