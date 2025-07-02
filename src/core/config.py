"""
Configuration management for AutoML Builder.

This module handles all configuration settings including environment variables,
feature flags, and runtime configurations.
"""

import os
from typing import Optional, Dict, Any, List, Literal
from pathlib import Path
from functools import lru_cache

from pydantic import BaseSettings, Field, validator, PostgresDsn, RedisDsn
from pydantic.networks import HttpUrl


class APIConfig(BaseSettings):
    """API server configuration."""

    host: str = Field("0.0.0.0", env="API_HOST")
    port: int = Field(8000, env="API_PORT")
    secret_key: str = Field(..., env="API_SECRET_KEY")
    debug: bool = Field(False, env="API_DEBUG")

    # CORS settings
    cors_origins: List[str] = Field(
        ["http://localhost:3000", "http://localhost:8501"], env="CORS_ORIGINS"
    )

    # Rate limiting
    rate_limit_enabled: bool = Field(True, env="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(100, env="RATE_LIMIT_REQUESTS")
    rate_limit_period: int = Field(60, env="RATE_LIMIT_PERIOD")  # seconds

    # JWT settings
    jwt_algorithm: str = Field("HS256", env="JWT_ALGORITHM")
    jwt_expiration_hours: int = Field(24, env="JWT_EXPIRATION_HOURS")

    class Config:
        env_prefix = "API_"


class OpenAIConfig(BaseSettings):
    """OpenAI API configuration."""

    api_key: str = Field(..., env="OPENAI_API_KEY")
    model: str = Field("gpt-4", env="OPENAI_MODEL")
    temperature: float = Field(0.7, env="OPENAI_TEMPERATURE")
    max_tokens: int = Field(2000, env="OPENAI_MAX_TOKENS")
    request_timeout: int = Field(60, env="OPENAI_REQUEST_TIMEOUT")

    # Model-specific settings
    embedding_model: str = Field("text-embedding-ada-002", env="OPENAI_EMBEDDING_MODEL")

    class Config:
        env_prefix = "OPENAI_"


class DatabaseConfig(BaseSettings):
    """Database configuration."""

    url: PostgresDsn = Field(..., env="DATABASE_URL")
    pool_size: int = Field(20, env="DATABASE_POOL_SIZE")
    max_overflow: int = Field(40, env="DATABASE_MAX_OVERFLOW")
    pool_timeout: int = Field(30, env="DATABASE_POOL_TIMEOUT")
    echo: bool = Field(False, env="DATABASE_ECHO")

    # Redis configuration
    redis_url: RedisDsn = Field("redis://localhost:6379", env="REDIS_URL")
    redis_ttl: int = Field(3600, env="REDIS_TTL")  # seconds
    redis_max_connections: int = Field(50, env="REDIS_MAX_CONNECTIONS")

    class Config:
        env_prefix = "DATABASE_"


class MLflowConfig(BaseSettings):
    """MLflow configuration."""

    tracking_uri: HttpUrl = Field("http://localhost:5000", env="MLFLOW_TRACKING_URI")
    artifact_root: str = Field("./mlruns", env="MLFLOW_ARTIFACT_ROOT")
    experiment_name: str = Field("automl-experiments", env="MLFLOW_EXPERIMENT_NAME")

    # MLflow server settings
    backend_store_uri: Optional[str] = Field(None, env="MLFLOW_BACKEND_STORE_URI")
    default_artifact_root: Optional[str] = Field(
        None, env="MLFLOW_DEFAULT_ARTIFACT_ROOT"
    )

    @validator("artifact_root")
    def validate_artifact_root(cls, v):
        """Ensure artifact root directory exists."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path.absolute())

    class Config:
        env_prefix = "MLFLOW_"


class OAuthConfig(BaseSettings):
    """OAuth configuration."""

    # Google OAuth
    google_client_id: Optional[str] = Field(None, env="GOOGLE_CLIENT_ID")
    google_client_secret: Optional[str] = Field(None, env="GOOGLE_CLIENT_SECRET")

    # GitHub OAuth
    github_client_id: Optional[str] = Field(None, env="GITHUB_CLIENT_ID")
    github_client_secret: Optional[str] = Field(None, env="GITHUB_CLIENT_SECRET")

    # OAuth redirect URIs
    redirect_uri: str = Field(
        "http://localhost:8000/auth/callback", env="OAUTH_REDIRECT_URI"
    )

    class Config:
        env_prefix = "OAUTH_"


class FeatureFlags(BaseSettings):
    """Feature flags configuration."""

    enable_debug_mode: bool = Field(True, env="ENABLE_DEBUG_MODE")
    enable_auto_mode: bool = Field(True, env="ENABLE_AUTO_MODE")
    enable_caching: bool = Field(True, env="ENABLE_CACHING")
    enable_async_processing: bool = Field(True, env="ENABLE_ASYNC_PROCESSING")
    enable_model_explainability: bool = Field(True, env="ENABLE_MODEL_EXPLAINABILITY")
    enable_advanced_preprocessing: bool = Field(
        True, env="ENABLE_ADVANCED_PREPROCESSING"
    )
    enable_ensemble_models: bool = Field(True, env="ENABLE_ENSEMBLE_MODELS")

    class Config:
        env_prefix = "FEATURE_"


class FileConfig(BaseSettings):
    """File handling configuration."""

    max_upload_size_mb: int = Field(100, env="MAX_UPLOAD_SIZE_MB")
    allowed_extensions: List[str] = Field(
        [".csv", ".xlsx", ".xls", ".json", ".parquet"], env="ALLOWED_FILE_EXTENSIONS"
    )

    # Directory paths
    upload_dir: str = Field("./data/uploads", env="UPLOAD_DIR")
    processed_dir: str = Field("./data/processed", env="PROCESSED_DIR")
    artifacts_dir: str = Field("./data/artifacts", env="ARTIFACTS_DIR")
    temp_dir: str = Field("./data/temp", env="TEMP_DIR")

    # File retention
    upload_retention_hours: int = Field(24, env="UPLOAD_RETENTION_HOURS")
    artifact_retention_days: int = Field(30, env="ARTIFACT_RETENTION_DAYS")

    @validator("upload_dir", "processed_dir", "artifacts_dir", "temp_dir")
    def create_directories(cls, v):
        """Ensure directories exist."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path.absolute())

    class Config:
        env_prefix = "FILE_"


class ModelConfig(BaseSettings):
    """Machine learning model configuration."""

    # Training settings
    cv_folds: int = Field(5, env="MODEL_CV_FOLDS")
    test_size: float = Field(0.2, env="MODEL_TEST_SIZE")
    random_state: int = Field(42, env="MODEL_RANDOM_STATE")
    n_jobs: int = Field(-1, env="MODEL_N_JOBS")

    # Model selection
    classification_models: List[str] = Field(
        ["logistic_regression", "random_forest", "xgboost", "lightgbm", "catboost"],
        env="CLASSIFICATION_MODELS",
    )
    regression_models: List[str] = Field(
        ["linear_regression", "random_forest", "xgboost", "lightgbm", "catboost"],
        env="REGRESSION_MODELS",
    )

    # Optimization settings
    optuna_n_trials: int = Field(100, env="OPTUNA_N_TRIALS")
    optuna_timeout: Optional[int] = Field(3600, env="OPTUNA_TIMEOUT")  # seconds
    optuna_n_jobs: int = Field(1, env="OPTUNA_N_JOBS")
    optuna_study_name: str = Field("automl_optimization", env="OPTUNA_STUDY_NAME")

    # Early stopping
    early_stopping_rounds: int = Field(10, env="EARLY_STOPPING_ROUNDS")
    early_stopping_tolerance: float = Field(0.001, env="EARLY_STOPPING_TOLERANCE")

    class Config:
        env_prefix = "MODEL_"


class AgentConfig(BaseSettings):
    """Agent system configuration."""

    # Agent execution
    max_iterations: int = Field(20, env="AGENT_MAX_ITERATIONS")
    timeout_seconds: int = Field(300, env="AGENT_TIMEOUT_SECONDS")
    retry_attempts: int = Field(3, env="AGENT_RETRY_ATTEMPTS")
    retry_delay: float = Field(1.0, env="AGENT_RETRY_DELAY")

    # Memory settings
    conversation_memory_limit: int = Field(50, env="CONVERSATION_MEMORY_LIMIT")
    agent_memory_ttl: int = Field(3600, env="AGENT_MEMORY_TTL")  # seconds

    # Execution modes
    execution_mode: Literal["sequential", "parallel"] = Field(
        "sequential", env="AGENT_EXECUTION_MODE"
    )
    enable_human_feedback: bool = Field(True, env="ENABLE_HUMAN_FEEDBACK")
    human_feedback_timeout: int = Field(120, env="HUMAN_FEEDBACK_TIMEOUT")  # seconds

    # Debug settings
    verbose_logging: bool = Field(False, env="AGENT_VERBOSE_LOGGING")
    trace_execution: bool = Field(False, env="AGENT_TRACE_EXECUTION")

    class Config:
        env_prefix = "AGENT_"


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    level: str = Field("INFO", env="LOG_LEVEL")
    format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT"
    )
    file_path: Optional[str] = Field(None, env="LOG_FILE_PATH")
    max_file_size: int = Field(10485760, env="LOG_MAX_FILE_SIZE")  # 10MB
    backup_count: int = Field(5, env="LOG_BACKUP_COUNT")

    # Structured logging
    json_format: bool = Field(True, env="LOG_JSON_FORMAT")
    include_context: bool = Field(True, env="LOG_INCLUDE_CONTEXT")

    class Config:
        env_prefix = "LOG_"


class Settings(BaseSettings):
    """Main settings class that combines all configurations."""

    # Environment
    environment: Literal["development", "staging", "production"] = Field(
        "development", env="ENVIRONMENT"
    )

    # Sub-configurations
    api: APIConfig = Field(default_factory=APIConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    oauth: OAuthConfig = Field(default_factory=OAuthConfig)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    files: FileConfig = Field(default_factory=FileConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    agents: AgentConfig = Field(default_factory=AgentConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Project metadata
    project_name: str = Field("AutoML Builder", env="PROJECT_NAME")
    version: str = Field("1.0.0", env="PROJECT_VERSION")

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"

    def get_database_url(self, async_mode: bool = False) -> str:
        """Get database URL with optional async support."""
        url = str(self.database.url)
        if async_mode and url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://")
        return url

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary (excluding sensitive data)."""
        data = self.dict()
        # Remove sensitive information
        sensitive_keys = ["secret_key", "api_key", "client_secret", "password"]

        def remove_sensitive(d: Dict[str, Any]) -> Dict[str, Any]:
            cleaned = {}
            for k, v in d.items():
                if any(sensitive in k.lower() for sensitive in sensitive_keys):
                    cleaned[k] = "***REDACTED***"
                elif isinstance(v, dict):
                    cleaned[k] = remove_sensitive(v)
                else:
                    cleaned[k] = v
            return cleaned

        return remove_sensitive(data)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings: Application settings
    """
    return Settings()


# Convenience function for accessing settings
settings = get_settings()


# Environment-specific configuration loading
def load_config_file(environment: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file based on environment.

    Args:
        environment: Environment name (development, staging, production)

    Returns:
        Dict containing configuration values
    """
    import yaml

    config_path = Path(f"configs/{environment}.yaml")
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


# Export commonly used settings
__all__ = [
    "Settings",
    "get_settings",
    "settings",
    "APIConfig",
    "OpenAIConfig",
    "DatabaseConfig",
    "MLflowConfig",
    "OAuthConfig",
    "FeatureFlags",
    "FileConfig",
    "ModelConfig",
    "AgentConfig",
    "LoggingConfig",
]
