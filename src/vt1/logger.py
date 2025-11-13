"""
Logging configuration for VT1 project.
Provides structured logging similar to SLF4J in Java.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

try:
    from PyQt6.QtCore import QObject, pyqtSignal

    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QObject = object
    pyqtSignal = None

# ANSI color codes for console output (optional)
COLORS = {
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[35m",  # Magenta
    "RESET": "\033[0m",  # Reset
}


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to console output."""

    def format(self, record):
        if hasattr(sys.stderr, "isatty") and sys.stderr.isatty():
            # Only use colors in terminal
            levelname = record.levelname
            if levelname in COLORS:
                record.levelname = f"{COLORS[levelname]}{levelname}{COLORS['RESET']}"
        return super().format(record)


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    use_colors: bool = False,
) -> logging.Logger:
    """
    Configure logger with console and optional file output.

    Args:
        name: Logger name (typically __name__)
        log_file: Optional file path for log output
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        use_colors: Whether to use colored output in console

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if use_colors:
        console_format = ColoredFormatter("[%(levelname)s] %(name)s - %(message)s")
    else:
        console_format = logging.Formatter("[%(levelname)s] %(name)s - %(message)s")

    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # Optional file handler with detailed format
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def configure_root_logger(log_dir: Optional[Path] = None, level: int = logging.INFO):
    """
    Configure the root logger for the entire application.

    Args:
        log_dir: Directory for log files (will create app.log)
        level: Logging level for the root logger
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # File handler
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "app.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)


def ensure_root_logger(log_dir: Optional[Path] = None, level: int = logging.INFO):
    """Ensure root logger has a file handler; idempotent."""
    root = logging.getLogger()

    # Derive log directory if not provided
    if log_dir is None:
        import sys
        import os

        frozen = getattr(sys, "frozen", False)

        if frozen:
            # Running as exe: use AppData/Local/vt1/logs
            app_data = Path(
                os.environ.get("LOCALAPPDATA", os.path.expanduser("~/.local"))
            )
            log_dir = app_data / "vt1" / "logs"
        else:
            # Dev mode: logger.py is at repo_root/src/vt1/logger.py
            # parents[0] = src/vt1
            # parents[1] = src
            # parents[2] = repo_root (D:\WORK\VT1)
            repo_root = Path(__file__).resolve().parents[2]
            log_dir = repo_root / "logs"

    # Check if we already have a FileHandler pointing to app.log
    log_file = log_dir / "app.log"
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler):
            if Path(handler.baseFilename).resolve() == log_file.resolve():
                return  # Already configured

    # Create directory and add file handler
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)
    root.addHandler(file_handler)

    if root.level > level:
        root.setLevel(level)

    print(f"✓ Root logger configured with file handler: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


if PYQT_AVAILABLE:

    class QtLogHandler(logging.Handler, QObject):
        """
        Log handler that emits log messages to PyQt widgets.

        Usage:
            qt_handler = QtLogHandler()
            qt_handler.log_signal.connect(lambda msg: text_widget.append(msg))
            logging.getLogger().addHandler(qt_handler)
        """

        log_signal = pyqtSignal(str)

        def __init__(self, parent=None):
            logging.Handler.__init__(self)
            QObject.__init__(self, parent)

        def emit(self, record):
            try:
                msg = self.format(record)
                self.log_signal.emit(msg)
            except Exception:
                self.handleError(record)

else:
    QtLogHandler = None
