# Shared logging setup for lmm-serving.

import logging
import os
import sys

_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
_configured = False

_RESET = "\033[0m"
_LEVEL_COLORS = {
    logging.DEBUG: "\033[36m",  # cyan
    logging.INFO: "\033[32m",  # green
    logging.WARNING: "\033[33m",  # yellow
    logging.ERROR: "\033[31m",  # red
    logging.CRITICAL: "\033[1;31m",  # bold red
}


class _ColorFormatter(logging.Formatter):

    def format(self, record):
        color = _LEVEL_COLORS.get(record.levelno)
        if color:
            original = record.levelname
            record.levelname = f"{color}{original}{_RESET}"
            try:
                return super().format(record)
            finally:
                record.levelname = original
        return super().format(record)


def _use_color():
    if os.environ.get("NO_COLOR"):
        return False
    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


def get_logger(name):
    # Return a logger with the shared lmm-serving format.
    global _configured
    if not _configured:
        requested = os.environ.get("LMM_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, requested, None)
        invalid = requested if not isinstance(level, int) else None
        if invalid:
            level = logging.INFO
        logging.basicConfig(level=level, format=_FORMAT, force=True)
        if _use_color():
            formatter = _ColorFormatter(_FORMAT)
            for handler in logging.getLogger().handlers:
                handler.setFormatter(formatter)
        if invalid:
            logging.getLogger("lmm").warning(
                "Invalid LMM_LOG_LEVEL=%r; defaulting to INFO", invalid
            )
        _configured = True
    return logging.getLogger(name)
