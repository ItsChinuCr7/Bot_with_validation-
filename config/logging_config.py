# ============================================================
# config/logging_config.py
# Central logging setup.
# Import setup_logging() once in app.py and every module
# automatically logs to both console and logs/app.log
# ============================================================

import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR  = os.path.join(os.path.dirname(__file__), "..", "logs")
LOG_FILE = os.path.join(LOG_DIR, "app.log")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FORMAT  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO"):
    """
    Call once at app startup.
    After this, any module can do:
        import logging
        logger = logging.getLogger(__name__)
        logger.info("message")
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Root logger
    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Avoid duplicate handlers on Streamlit reruns
    if root.handlers:
        return

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # ── Console handler ────────────────────────────────────
    console = logging.StreamHandler()
    console.setLevel(numeric_level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # ── File handler (rotating, max 5MB, keep 3 backups) ──
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes    = 5 * 1024 * 1024,
        backupCount = 3,
        encoding    = "utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Silence noisy third-party loggers
    for noisy in ["httpx", "httpcore", "urllib3", "selenium", "WDM", "faiss"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.info("=" * 60)
    logging.info("AI Document Intelligence Assistant — started")
    logging.info("=" * 60)