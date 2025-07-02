# src/utils/logger.py
import logging
import sys
import structlog
from typing import Any, Dict, Optional
from datetime import datetime
import json
from pathlib import Path
from src.core.config import settings

# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)


def setup_logging():
    """Configure structured logging for the application"""

    # Configure standard logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(message)s",
        stream=sys.stdout,
    )

    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


class Logger:
    """Custom logger wrapper with additional functionality"""

    def __init__(self, name: str):
        self.logger = structlog.get_logger(name)
        self._context: Dict[str, Any] = {}

    def bind(self, **kwargs) -> "Logger":
        """Add context to logger"""
        self._context.update(kwargs)
        self.logger = self.logger.bind(**kwargs)
        return self

    def unbind(self, *keys) -> "Logger":
        """Remove context from logger"""
        for key in keys:
            self._context.pop(key, None)
        self.logger = self.logger.unbind(*keys)
        return self

    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, **self._merge_context(kwargs))

    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, **self._merge_context(kwargs))

    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, **self._merge_context(kwargs))

    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, **self._merge_context(kwargs))

    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        self.logger.exception(message, **self._merge_context(kwargs))

    def _merge_context(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge permanent context with temporary kwargs"""
        return {**self._context, **kwargs}

    def log_api_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        user_id: Optional[str] = None,
        **kwargs,
    ):
        """Log API request with standard fields"""
        self.info(
            "API Request",
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=duration_ms,
            user_id=user_id,
            **kwargs,
        )

    def log_agent_action(
        self,
        agent_name: str,
        action: str,
        state: Dict[str, Any],
        result: Any = None,
        error: Optional[str] = None,
        **kwargs,
    ):
        """Log agent action with context"""
        log_data = {
            "agent_name": agent_name,
            "action": action,
            "state_keys": list(state.keys()),
            **kwargs,
        }

        if result is not None:
            log_data["result_type"] = type(result).__name__

        if error:
            log_data["error"] = error
            self.error(f"Agent action failed: {agent_name}.{action}", **log_data)
        else:
            self.info(f"Agent action: {agent_name}.{action}", **log_data)

    def log_ml_experiment(
        self,
        experiment_id: str,
        model_name: str,
        metrics: Dict[str, float],
        parameters: Dict[str, Any],
        **kwargs,
    ):
        """Log ML experiment details"""
        self.info(
            "ML Experiment",
            experiment_id=experiment_id,
            model_name=model_name,
            metrics=metrics,
            parameters=parameters,
            **kwargs,
        )

    def log_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs,
    ):
        """Log a generic event"""
        self.info(
            f"Event: {event_type}",
            event_type=event_type,
            event_data=event_data,
            session_id=session_id,
            user_id=user_id,
            **kwargs,
        )


# Create a global logger instance
def get_logger(name: str) -> Logger:
    """Get a logger instance for a module"""
    return Logger(name)


# Setup logging on module import
setup_logging()


# File handler for persistent logs
class FileHandler:
    """Custom file handler for structured logs"""

    def __init__(self, filename: str = "logs/automl.log", max_size_mb: int = 100):
        self.filename = filename
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.file_path = Path(filename)
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Ensure log file and directory exist"""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.touch(exist_ok=True)

    def _rotate_if_needed(self):
        """Rotate log file if it exceeds max size"""
        if self.file_path.stat().st_size > self.max_size_bytes:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rotated_path = self.file_path.with_suffix(f".{timestamp}.log")
            self.file_path.rename(rotated_path)
            self._ensure_file_exists()

    def write_log(self, log_entry: Dict[str, Any]):
        """Write log entry to file"""
        self._rotate_if_needed()

        with open(self.file_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")


# Global file handler
file_handler = FileHandler()


# Utility function for timing operations
class Timer:
    """Context manager for timing operations"""

    def __init__(self, logger: Logger, operation_name: str):
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (datetime.now() - self.start_time).total_seconds() * 1000

        if exc_type:
            self.logger.error(
                f"Operation failed: {self.operation_name}",
                duration_ms=duration_ms,
                error=str(exc_val),
            )
        else:
            self.logger.debug(
                f"Operation completed: {self.operation_name}", duration_ms=duration_ms
            )
