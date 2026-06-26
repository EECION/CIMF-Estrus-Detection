import logging
import sys
from enum import Enum
from typing import Callable, Optional


class LogLevel(Enum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    SUCCESS = 4


_LEVEL_MAP = {
    LogLevel.DEBUG: logging.DEBUG,
    LogLevel.INFO: logging.INFO,
    LogLevel.WARNING: logging.WARNING,
    LogLevel.ERROR: logging.ERROR,
    LogLevel.SUCCESS: logging.INFO,
}


class UnifiedLogger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._level = LogLevel.INFO
        self._gui_callback: Optional[Callable[[str, str], None]] = None
        self._logger = logging.getLogger("CIMFTrainer")
        self._logger.setLevel(logging.DEBUG)
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self._logger.addHandler(handler)

    def set_level(self, level: LogLevel):
        self._level = level

    def set_gui_callback(self, callback: Callable[[str, str], None]):
        self._gui_callback = callback

    def _emit(self, level: LogLevel, message: str, source: str = ""):
        if level.value < self._level.value:
            return
        prefix = f"[{source}] " if source else ""
        full_message = f"{prefix}{message}"
        log_level = _LEVEL_MAP.get(level, logging.INFO)
        self._logger.log(log_level, full_message)
        if self._gui_callback:
            gui_level = level.name if level != LogLevel.SUCCESS else "SUCCESS"
            try:
                self._gui_callback(full_message, gui_level)
            except Exception:
                pass

    def debug(self, message: str, source: str = ""):
        self._emit(LogLevel.DEBUG, message, source)

    def info(self, message: str, source: str = ""):
        self._emit(LogLevel.INFO, message, source)

    def warning(self, message: str, source: str = ""):
        self._emit(LogLevel.WARNING, message, source)

    def error(self, message: str, source: str = ""):
        self._emit(LogLevel.ERROR, message, source)

    def success(self, message: str, source: str = ""):
        self._emit(LogLevel.SUCCESS, message, source)


def get_logger() -> UnifiedLogger:
    return UnifiedLogger()


def set_log_level(level: LogLevel):
    get_logger().set_level(level)


def setup_gui_logging(callback: Callable[[str, str], None]):
    get_logger().set_gui_callback(callback)
