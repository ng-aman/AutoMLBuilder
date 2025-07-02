# src/utils/validators.py
import re
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np


def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def validate_password(password: str, min_length: int = 8) -> tuple[bool, Optional[str]]:
    """
    Validate password strength

    Returns:
        (is_valid, error_message)
    """
    if len(password) < min_length:
        return False, f"Password must be at least {min_length} characters long"

    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"

    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter"

    if not re.search(r"\d", password):
        return False, "Password must contain at least one number"

    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character"

    return True, None


def validate_file_type(filename: str, allowed_extensions: List[str]) -> bool:
    """Validate file extension"""
    file_ext = Path(filename).suffix.lower()
    return file_ext in allowed_extensions


def validate_file_size(file_size: int, max_size_mb: int) -> bool:
    """Validate file size"""
    max_size_bytes = max_size_mb * 1024 * 1024
    return file_size <= max_size_bytes


def validate_dataset_format(file_path: str) -> tuple[bool, Optional[str]]:
    """
    Validate dataset format and structure

    Returns:
        (is_valid, error_message)
    """
    try:
        file_ext = Path(file_path).suffix.lower()

        # Load dataset
        if file_ext == ".csv":
            df = pd.read_csv(file_path, nrows=10)
        elif file_ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path, nrows=10)
        elif file_ext == ".json":
            df = pd.read_json(file_path)
            df = df.head(10)
        else:
            return False, f"Unsupported file type: {file_ext}"

        # Basic validations
        if df.empty:
            return False, "Dataset is empty"

        if len(df.columns) < 2:
            return False, "Dataset must have at least 2 columns"

        if len(df) < 10:
            return False, "Dataset must have at least 10 rows"

        # Check for all null columns
        null_columns = df.columns[df.isnull().all()].tolist()
        if null_columns:
            return False, f"Columns with all null values: {', '.join(null_columns)}"

        return True, None

    except Exception as e:
        return False, f"Error reading dataset: {str(e)}"


def validate_column_name(name: str) -> bool:
    """Validate column name format"""
    # Allow alphanumeric, underscore, and hyphen
    pattern = r"^[a-zA-Z0-9_-]+$"
    return bool(re.match(pattern, name))


def validate_target_variable(
    df: pd.DataFrame, target_column: str, problem_type: str
) -> tuple[bool, Optional[str]]:
    """
    Validate target variable for the problem type

    Returns:
        (is_valid, error_message)
    """
    if target_column not in df.columns:
        return False, f"Target column '{target_column}' not found in dataset"

    target_series = df[target_column]

    # Check for null values
    null_count = target_series.isnull().sum()
    if null_count > 0:
        return False, f"Target variable has {null_count} missing values"

    if problem_type == "classification":
        # Check number of unique values
        n_unique = target_series.nunique()
        if n_unique < 2:
            return False, "Classification target must have at least 2 unique values"

        if n_unique > 100:
            return (
                False,
                f"Target has {n_unique} unique values. Consider regression instead.",
            )

        # Check class balance
        value_counts = target_series.value_counts()
        min_class_size = value_counts.min()
        if min_class_size < 5:
            return (
                False,
                f"Smallest class has only {min_class_size} samples. Need at least 5.",
            )

    elif problem_type == "regression":
        # Check if numeric
        if not pd.api.types.is_numeric_dtype(target_series):
            return False, "Regression target must be numeric"

        # Check variance
        if target_series.std() == 0:
            return False, "Target variable has zero variance"

    return True, None


def validate_model_parameters(
    model_name: str, parameters: Dict[str, Any]
) -> tuple[bool, Optional[str]]:
    """
    Validate model hyperparameters

    Returns:
        (is_valid, error_message)
    """
    # Define parameter constraints for common models
    constraints = {
        "RandomForestClassifier": {
            "n_estimators": (1, 1000),
            "max_depth": (1, 100),
            "min_samples_split": (2, 100),
            "min_samples_leaf": (1, 100),
        },
        "XGBClassifier": {
            "n_estimators": (1, 1000),
            "max_depth": (1, 20),
            "learning_rate": (0.001, 1.0),
            "subsample": (0.1, 1.0),
        },
        "LogisticRegression": {"C": (0.001, 1000), "max_iter": (100, 10000)},
    }

    if model_name not in constraints:
        return True, None  # No validation for unknown models

    model_constraints = constraints[model_name]

    for param, value in parameters.items():
        if param in model_constraints:
            min_val, max_val = model_constraints[param]
            if not (min_val <= value <= max_val):
                return False, f"{param} must be between {min_val} and {max_val}"

    return True, None


def validate_api_key(api_key: str, provider: str = "openai") -> bool:
    """Validate API key format"""
    patterns = {
        "openai": r"^sk-[a-zA-Z0-9]{48}$",
        "google": r"^[a-zA-Z0-9_-]{39}$",
    }

    pattern = patterns.get(provider)
    if not pattern:
        return len(api_key) > 10  # Basic length check

    return bool(re.match(pattern, api_key))


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    # Remove path components
    filename = Path(filename).name

    # Replace special characters
    filename = re.sub(r"[^\w\s.-]", "_", filename)

    # Limit length
    max_length = 255
    if len(filename) > max_length:
        name, ext = filename.rsplit(".", 1)
        filename = name[: max_length - len(ext) - 1] + "." + ext

    return filename


def validate_session_id(session_id: str) -> bool:
    """Validate session ID format (UUID)"""
    uuid_pattern = r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$"
    return bool(re.match(uuid_pattern, session_id.lower()))


def validate_json_schema(
    data: Dict[str, Any], schema: Dict[str, Any]
) -> tuple[bool, Optional[str]]:
    """
    Validate JSON data against a schema

    Returns:
        (is_valid, error_message)
    """
    try:
        import jsonschema

        jsonschema.validate(instance=data, schema=schema)
        return True, None
    except jsonschema.ValidationError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Schema validation error: {str(e)}"
