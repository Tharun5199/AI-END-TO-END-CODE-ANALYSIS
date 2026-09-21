"""Centralized logging setup used across the whole project."""
import logging
import os
from datetime import datetime

from codeanalyzer.config import PROJECT_ROOT

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.log"

_FORMAT = "[%(asctime)s] %(levelname)-8s %(name)s - %(message)s"

_root = logging.getLogger()
if not getattr(_root, "_codeanalyzer_configured", False):
    _root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    _formatter = logging.Formatter(_FORMAT)

    _file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    _file_handler.setFormatter(_formatter)
    _root.addHandler(_file_handler)

    # Avoid a second console handler if something (e.g. Jupyter) already added one.
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in _root.handlers):
        _stream_handler = logging.StreamHandler()
        _stream_handler.setFormatter(_formatter)
        _root.addHandler(_stream_handler)

    # Third-party libraries that log every single HTTP request at INFO level.
    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "sentence_transformers",
                  "chromadb", "git"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _root._codeanalyzer_configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger configured with the project's format."""
    return logging.getLogger(name)
