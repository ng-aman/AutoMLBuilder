"""
AutoML Builder Preprocessing Package.

This package provides comprehensive data preprocessing capabilities including
data cleaning, encoding, and scaling for machine learning pipelines.
"""

# Version information
__version__ = "1.0.0"
__author__ = "AutoML Builder Team"
__description__ = "Advanced preprocessing toolkit for automated machine learning"

# Import cleaner module components
from .cleaner import (
    DataCleaner,
    MissingValueHandler,
    OutlierHandler,
    DuplicateHandler,
    DataTypeOptimizer,
    clean_dataset,
    handle_missing_values,
    detect_outliers,
    remove_duplicates,
)

# Import encoder module components
from .encoder import (
    AdvancedEncoder,
    TargetEncoder,
    FrequencyEncoder,
    BinaryEncoder,
    HashingEncoder,
    DateTimeEncoder,
    TextEncoder,
    auto_encode,
    encode_categorical,
    encode_datetime,
    encode_text,
)

# Import scaler module components
from .scaler import (
    AdvancedScaler,
    MultiScaler,
    OutlierRobustScaler,
    auto_scale,
    scale_for_model,
)

# Import types for better IDE support
from typing import Union, Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np


class PreprocessingPipeline:
    """
    Unified preprocessing pipeline that combines cleaning, encoding, and scaling.

    Provides a high-level interface for complete data preprocessing.
    """

    def __init__(
        self,
        cleaning_config: Optional[Dict[str, Any]] = None,
        encoding_config: Optional[Dict[str, Any]] = None,
        scaling_config: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ):
        """
        Initialize preprocessing pipeline.

        Args:
            cleaning_config: Configuration for data cleaning
            encoding_config: Configuration for encoding
            scaling_config: Configuration for scaling
            verbose: Whether to print progress messages
        """
        self.cleaning_config = cleaning_config or {}
        self.encoding_config = encoding_config or {}
        self.scaling_config = scaling_config or {}
        self.verbose = verbose

        # Initialize components
        self.cleaner = None
        self.encoder = None
        self.scaler = None

        # Track preprocessing steps
        self.preprocessing_steps = []
        self.feature_info = {}

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        feature_types: Optional[Dict[str, str]] = None,
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[Union[pd.Series, np.ndarray]]]:
        """
        Fit and transform data through complete preprocessing pipeline.

        Args:
            X: Input features
            y: Target variable (optional)
            feature_types: Optional feature type specifications

        Returns:
            Tuple of (preprocessed features, preprocessed target)
        """
        # Ensure DataFrame format for processing
        X_processed = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y_processed = y.copy() if y is not None else None

        # Step 1: Data Cleaning
        if self.verbose:
            print("Step 1: Data Cleaning...")

        self.cleaner = DataCleaner(**self.cleaning_config)
        cleaning_report = self.cleaner.analyze(X_processed)
        X_processed = self.cleaner.clean(X_processed)

        self.preprocessing_steps.append({"step": "cleaning", "report": cleaning_report})

        # Step 2: Encoding
        if self.verbose:
            print("Step 2: Feature Encoding...")

        self.encoder = AdvancedEncoder(**self.encoding_config)
        X_processed = self.encoder.fit_transform(
            X_processed, feature_types=feature_types
        )

        encoding_report = self.encoder.get_encoding_report()
        self.preprocessing_steps.append({"step": "encoding", "report": encoding_report})

        # Step 3: Scaling
        if self.verbose:
            print("Step 3: Feature Scaling...")

        # Determine which columns to scale (numeric only)
        numeric_columns = X_processed.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        if numeric_columns:
            self.scaler = AdvancedScaler(**self.scaling_config)
            X_processed[numeric_columns] = self.scaler.fit_transform(
                X_processed[numeric_columns]
            )

            scaling_report = self.scaler.get_scaling_report()
            self.preprocessing_steps.append(
                {"step": "scaling", "report": scaling_report}
            )

        # Handle target variable if provided
        if y_processed is not None and self.encoder:
            y_processed = self.encoder.encode_target(y_processed)

        # Store feature information
        self._store_feature_info(X_processed)

        if self.verbose:
            print("Preprocessing complete!")

        # Return in original format
        if isinstance(X, np.ndarray):
            X_processed = X_processed.values

        return X_processed, y_processed

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[Union[pd.Series, np.ndarray]]]:
        """
        Transform new data using fitted preprocessing pipeline.

        Args:
            X: Input features
            y: Target variable (optional)

        Returns:
            Tuple of (preprocessed features, preprocessed target)
        """
        if not all([self.cleaner, self.encoder]):
            raise ValueError("Pipeline not fitted. Call fit_transform first.")

        X_processed = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y_processed = y.copy() if y is not None else None

        # Apply cleaning
        X_processed = self.cleaner.clean(X_processed)

        # Apply encoding
        X_processed = self.encoder.transform(X_processed)

        # Apply scaling
        if self.scaler:
            numeric_columns = X_processed.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            if numeric_columns:
                X_processed[numeric_columns] = self.scaler.transform(
                    X_processed[numeric_columns]
                )

        # Handle target variable
        if y_processed is not None and self.encoder:
            y_processed = self.encoder.encode_target(y_processed)

        # Return in original format
        if isinstance(X, np.ndarray):
            X_processed = X_processed.values

        return X_processed, y_processed

    def _store_feature_info(self, X_processed: pd.DataFrame):
        """Store information about processed features."""
        self.feature_info = {
            "n_features": X_processed.shape[1],
            "feature_names": X_processed.columns.tolist(),
            "feature_types": X_processed.dtypes.to_dict(),
            "numeric_features": X_processed.select_dtypes(
                include=[np.number]
            ).columns.tolist(),
            "categorical_features": X_processed.select_dtypes(
                exclude=[np.number]
            ).columns.tolist(),
        }

    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get summary of all preprocessing steps."""
        return {
            "steps": self.preprocessing_steps,
            "feature_info": self.feature_info,
            "components": {
                "cleaner": self.cleaner is not None,
                "encoder": self.encoder is not None,
                "scaler": self.scaler is not None,
            },
        }

    def save_pipeline(self, filepath: str):
        """Save preprocessing pipeline to file."""
        import joblib

        pipeline_data = {
            "cleaner": self.cleaner,
            "encoder": self.encoder,
            "scaler": self.scaler,
            "feature_info": self.feature_info,
            "config": {
                "cleaning": self.cleaning_config,
                "encoding": self.encoding_config,
                "scaling": self.scaling_config,
            },
        }

        joblib.dump(pipeline_data, filepath)
        if self.verbose:
            print(f"Pipeline saved to {filepath}")

    def load_pipeline(self, filepath: str):
        """Load preprocessing pipeline from file."""
        import joblib

        pipeline_data = joblib.load(filepath)

        self.cleaner = pipeline_data["cleaner"]
        self.encoder = pipeline_data["encoder"]
        self.scaler = pipeline_data["scaler"]
        self.feature_info = pipeline_data["feature_info"]
        self.cleaning_config = pipeline_data["config"]["cleaning"]
        self.encoding_config = pipeline_data["config"]["encoding"]
        self.scaling_config = pipeline_data["config"]["scaling"]

        if self.verbose:
            print(f"Pipeline loaded from {filepath}")


# Convenience functions for quick preprocessing
def preprocess_data(
    X: Union[pd.DataFrame, np.ndarray],
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    task_type: str = "classification",
    handle_text: bool = True,
    handle_dates: bool = True,
    scale_features: bool = True,
    verbose: bool = True,
) -> Tuple[
    Union[pd.DataFrame, np.ndarray],
    Optional[Union[pd.Series, np.ndarray]],
    PreprocessingPipeline,
]:
    """
    Quick preprocessing function with sensible defaults.

    Args:
        X: Input features
        y: Target variable (optional)
        task_type: "classification" or "regression"
        handle_text: Whether to process text features
        handle_dates: Whether to process datetime features
        scale_features: Whether to scale numeric features
        verbose: Whether to print progress

    Returns:
        Tuple of (preprocessed features, preprocessed target, fitted pipeline)
    """
    # Configure pipeline based on options
    cleaning_config = {
        "missing_strategy": "auto",
        "outlier_method": "isolation_forest" if task_type == "regression" else None,
        "optimize_dtypes": True,
    }

    encoding_config = {
        "strategy": "auto",
        "handle_unknown": "ignore",
        "text_encoding": handle_text,
        "datetime_encoding": handle_dates,
    }

    scaling_config = (
        {"strategy": "auto", "handle_outliers": True} if scale_features else None
    )

    # Create and run pipeline
    pipeline = PreprocessingPipeline(
        cleaning_config=cleaning_config,
        encoding_config=encoding_config,
        scaling_config=scaling_config if scale_features else {},
        verbose=verbose,
    )

    X_processed, y_processed = pipeline.fit_transform(X, y)

    return X_processed, y_processed, pipeline


def get_preprocessing_recommendations(
    X: Union[pd.DataFrame, np.ndarray],
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    task_type: str = "classification",
) -> Dict[str, Any]:
    """
    Get preprocessing recommendations based on data analysis.

    Args:
        X: Input features
        y: Target variable (optional)
        task_type: "classification" or "regression"

    Returns:
        Dictionary with preprocessing recommendations
    """
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

    # Analyze data
    cleaner = DataCleaner()
    analysis = cleaner.analyze(X_df)

    recommendations = {
        "cleaning": {
            "missing_values": {
                "strategy": "auto",
                "reason": f"{analysis['missing_values']['total_missing']} missing values found",
            },
            "outliers": {
                "needed": analysis["outliers"]["total_outliers"] > 0,
                "method": "isolation_forest" if task_type == "regression" else "iqr",
            },
            "duplicates": {
                "needed": analysis["duplicates"]["n_duplicates"] > 0,
                "count": analysis["duplicates"]["n_duplicates"],
            },
        },
        "encoding": {
            "categorical_features": analysis["data_types"]["categorical_columns"],
            "high_cardinality": [
                col
                for col in analysis["data_types"]["categorical_columns"]
                if X_df[col].nunique() > 10
            ],
            "datetime_features": analysis["data_types"]["datetime_columns"],
            "text_features": analysis["data_types"].get("text_columns", []),
        },
        "scaling": {
            "needed": True,  # Generally recommended except for tree models
            "strategy": (
                "robust" if analysis["outliers"]["total_outliers"] > 0 else "standard"
            ),
        },
    }

    return recommendations


# Define preprocessing presets
PREPROCESSING_PRESETS = {
    "minimal": {
        "cleaning": {"missing_strategy": "drop", "optimize_dtypes": True},
        "encoding": {"strategy": "label"},
        "scaling": {},  # No scaling
    },
    "standard": {
        "cleaning": {"missing_strategy": "auto", "outlier_method": "iqr"},
        "encoding": {"strategy": "auto"},
        "scaling": {"strategy": "standard"},
    },
    "robust": {
        "cleaning": {"missing_strategy": "auto", "outlier_method": "isolation_forest"},
        "encoding": {"strategy": "auto", "handle_unknown": "ignore"},
        "scaling": {"strategy": "robust", "handle_outliers": True},
    },
    "advanced": {
        "cleaning": {
            "missing_strategy": "iterative",
            "outlier_method": "ensemble",
            "optimize_dtypes": True,
        },
        "encoding": {
            "strategy": "auto",
            "handle_unknown": "ignore",
            "text_encoding": True,
            "datetime_encoding": True,
        },
        "scaling": {"strategy": "auto", "handle_outliers": True},
    },
}


def create_preprocessing_pipeline(
    preset: str = "standard", **kwargs
) -> PreprocessingPipeline:
    """
    Create preprocessing pipeline from preset.

    Args:
        preset: Preset name ("minimal", "standard", "robust", "advanced")
        **kwargs: Additional configuration overrides

    Returns:
        Configured preprocessing pipeline
    """
    if preset not in PREPROCESSING_PRESETS:
        raise ValueError(
            f"Unknown preset: {preset}. Choose from {list(PREPROCESSING_PRESETS.keys())}"
        )

    config = PREPROCESSING_PRESETS[preset].copy()

    # Apply overrides
    for key, value in kwargs.items():
        if key in config:
            config[key].update(value)

    return PreprocessingPipeline(**config)


# Export all public components
__all__ = [
    # Main pipeline
    "PreprocessingPipeline",
    # Cleaner components
    "DataCleaner",
    "MissingValueHandler",
    "OutlierHandler",
    "DuplicateHandler",
    "DataTypeOptimizer",
    "clean_dataset",
    "handle_missing_values",
    "detect_outliers",
    "remove_duplicates",
    # Encoder components
    "AdvancedEncoder",
    "TargetEncoder",
    "FrequencyEncoder",
    "BinaryEncoder",
    "HashingEncoder",
    "DateTimeEncoder",
    "TextEncoder",
    "auto_encode",
    "encode_categorical",
    "encode_datetime",
    "encode_text",
    # Scaler components
    "AdvancedScaler",
    "MultiScaler",
    "OutlierRobustScaler",
    "auto_scale",
    "scale_for_model",
    # Convenience functions
    "preprocess_data",
    "get_preprocessing_recommendations",
    "create_preprocessing_pipeline",
    # Presets
    "PREPROCESSING_PRESETS",
]
