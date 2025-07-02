# src/utils/exceptions.py
from typing import Optional, Dict, Any
from fastapi import HTTPException, status


class AutoMLException(Exception):
    """Base exception for AutoML application"""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary"""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
        }


# API Exceptions
class APIException(HTTPException):
    """Base API exception"""

    def __init__(
        self,
        status_code: int,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        detail = {
            "error": error_code or self.__class__.__name__,
            "message": message,
            "details": details or {},
        }
        super().__init__(status_code=status_code, detail=detail)


class AuthenticationError(APIException):
    """Authentication failed"""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            message=message,
            error_code="AUTHENTICATION_ERROR",
        )


class AuthorizationError(APIException):
    """Authorization failed"""

    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            message=message,
            error_code="AUTHORIZATION_ERROR",
        )


class ResourceNotFoundError(APIException):
    """Resource not found"""

    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            message=f"{resource_type} not found",
            error_code="RESOURCE_NOT_FOUND",
            details={"resource_type": resource_type, "resource_id": resource_id},
        )


class ValidationError(APIException):
    """Validation error"""

    def __init__(self, message: str, field: Optional[str] = None):
        details = {}
        if field:
            details["field"] = field

        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            message=message,
            error_code="VALIDATION_ERROR",
            details=details,
        )


class FileTooLargeError(APIException):
    """File size exceeds limit"""

    def __init__(self, file_size: int, max_size: int):
        super().__init__(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            message=f"File size ({file_size} bytes) exceeds maximum allowed size ({max_size} bytes)",
            error_code="FILE_TOO_LARGE",
            details={"file_size": file_size, "max_size": max_size},
        )


class InvalidFileTypeError(APIException):
    """Invalid file type"""

    def __init__(self, file_type: str, allowed_types: list):
        super().__init__(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            message=f"File type '{file_type}' is not supported",
            error_code="INVALID_FILE_TYPE",
            details={"file_type": file_type, "allowed_types": allowed_types},
        )


# Agent Exceptions
class AgentError(AutoMLException):
    """Base agent exception"""

    def __init__(
        self, agent_name: str, message: str, details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=f"Agent '{agent_name}' error: {message}",
            error_code=f"{agent_name.upper()}_AGENT_ERROR",
            details={**(details or {}), "agent": agent_name},
        )


class AgentTimeoutError(AgentError):
    """Agent execution timeout"""

    def __init__(self, agent_name: str, timeout_seconds: int):
        super().__init__(
            agent_name=agent_name,
            message=f"Execution timed out after {timeout_seconds} seconds",
            details={"timeout_seconds": timeout_seconds},
        )


class AgentExecutionError(AgentError):
    """Agent execution failed"""

    def __init__(self, agent_name: str, error: str):
        super().__init__(
            agent_name=agent_name,
            message=f"Execution failed: {error}",
            details={"error": error},
        )


# ML Exceptions
class MLError(AutoMLException):
    """Base ML exception"""

    pass


class DatasetError(MLError):
    """Dataset related error"""

    def __init__(self, message: str, dataset_id: Optional[str] = None):
        details = {}
        if dataset_id:
            details["dataset_id"] = dataset_id

        super().__init__(message=message, error_code="DATASET_ERROR", details=details)


class ModelTrainingError(MLError):
    """Model training failed"""

    def __init__(self, model_name: str, error: str):
        super().__init__(
            message=f"Training '{model_name}' failed: {error}",
            error_code="MODEL_TRAINING_ERROR",
            details={"model_name": model_name, "error": error},
        )


class PreprocessingError(MLError):
    """Data preprocessing failed"""

    def __init__(self, step: str, error: str):
        super().__init__(
            message=f"Preprocessing step '{step}' failed: {error}",
            error_code="PREPROCESSING_ERROR",
            details={"step": step, "error": error},
        )


class OptimizationError(MLError):
    """Hyperparameter optimization failed"""

    def __init__(self, error: str, study_name: Optional[str] = None):
        details = {"error": error}
        if study_name:
            details["study_name"] = study_name

        super().__init__(
            message=f"Optimization failed: {error}",
            error_code="OPTIMIZATION_ERROR",
            details=details,
        )


# Workflow Exceptions
class WorkflowError(AutoMLException):
    """Workflow execution error"""

    pass


class StateError(WorkflowError):
    """Invalid state error"""

    def __init__(self, message: str, current_state: Dict[str, Any]):
        super().__init__(
            message=message,
            error_code="STATE_ERROR",
            details={"current_state": current_state},
        )


class CheckpointError(WorkflowError):
    """Checkpoint operation failed"""

    def __init__(self, operation: str, error: str):
        super().__init__(
            message=f"Checkpoint {operation} failed: {error}",
            error_code="CHECKPOINT_ERROR",
            details={"operation": operation, "error": error},
        )


# Utility function to handle exceptions
def handle_exception(logger, exception: Exception) -> Dict[str, Any]:
    """Handle exception and return error response"""
    if isinstance(exception, AutoMLException):
        logger.error(
            exception.message,
            error_code=exception.error_code,
            details=exception.details,
        )
        return exception.to_dict()

    elif isinstance(exception, APIException):
        logger.error(str(exception.detail), status_code=exception.status_code)
        raise exception

    else:
        # Unknown exception
        logger.exception("Unexpected error occurred")
        return {
            "error": "INTERNAL_ERROR",
            "message": "An unexpected error occurred",
            "details": {"error_type": type(exception).__name__},
        }
