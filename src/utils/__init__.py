# src/utils/__init__.py
"""
AutoML Builder Utilities Module

Common utilities for logging, exception handling, and validation.
"""

from src.utils.logger import get_logger, Logger, Timer
from src.utils.exceptions import (
    AutoMLException,
    APIException,
    AuthenticationError,
    AuthorizationError,
    ResourceNotFoundError,
    ValidationError,
    FileTooLargeError,
    InvalidFileTypeError,
    AgentError,
    AgentTimeoutError,
    AgentExecutionError,
    MLError,
    DatasetError,
    ModelTrainingError,
    PreprocessingError,
    OptimizationError,
    WorkflowError,
    StateError,
    CheckpointError,
    handle_exception,
)

__all__ = [
    # Logging
    "get_logger",
    "Logger",
    "Timer",
    # Base exceptions
    "AutoMLException",
    "APIException",
    # API exceptions
    "AuthenticationError",
    "AuthorizationError",
    "ResourceNotFoundError",
    "ValidationError",
    "FileTooLargeError",
    "InvalidFileTypeError",
    # Agent exceptions
    "AgentError",
    "AgentTimeoutError",
    "AgentExecutionError",
    # ML exceptions
    "MLError",
    "DatasetError",
    "ModelTrainingError",
    "PreprocessingError",
    "OptimizationError",
    # Workflow exceptions
    "WorkflowError",
    "StateError",
    "CheckpointError",
    # Exception handler
    "handle_exception",
]
