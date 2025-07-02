# src/ml/preprocessing/scaler.py
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union, Literal
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    QuantileTransformer,
    PowerTransformer,
    Normalizer,
)
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
from scipy import stats
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AdvancedScaler:
    """
    Advanced data scaling and normalization handler.

    Provides intelligent scaling selection, outlier-robust scaling,
    and feature-specific scaling strategies.
    """

    def __init__(self, strategy: str = "auto", handle_outliers: bool = True):
        """
        Initialize advanced scaler.

        Args:
            strategy: Scaling strategy ("auto", "standard", "minmax", "robust", etc.)
            handle_outliers: Whether to use outlier-robust methods
        """
        self.strategy = strategy
        self.handle_outliers = handle_outliers
        self.scalers = {}
        self.scaling_info = {}
        self.feature_distributions = {}

        # Available scaling methods
        self.available_scalers = {
            "standard": StandardScaler,
            "minmax": MinMaxScaler,
            "robust": RobustScaler,
            "maxabs": MaxAbsScaler,
            "quantile_normal": lambda: QuantileTransformer(
                output_distribution="normal"
            ),
            "quantile_uniform": lambda: QuantileTransformer(
                output_distribution="uniform"
            ),
            "power_yeo": lambda: PowerTransformer(method="yeo-johnson"),
            "power_box": lambda: PowerTransformer(method="box-cox"),
            "normalizer": Normalizer,
        }

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        feature_types: Optional[Dict[str, str]] = None,
        exclude_columns: Optional[List[str]] = None,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Fit scalers and transform data.

        Args:
            X: Input data
            feature_types: Optional dict mapping features to types
            exclude_columns: Columns to exclude from scaling

        Returns:
            Scaled data
        """
        # Convert to DataFrame for easier handling
        X_df = self._ensure_dataframe(X)
        exclude_columns = exclude_columns or []

        # Analyze features if auto strategy
        if self.strategy == "auto":
            self._analyze_features(X_df, exclude_columns)

        # Apply scaling
        X_scaled = X_df.copy()

        for col in X_df.columns:
            if col in exclude_columns:
                continue

            # Determine scaling method for this column
            if self.strategy == "auto":
                scaler_name = self._select_scaler_for_feature(col)
            else:
                scaler_name = self.strategy

            # Create and fit scaler
            scaler = self._create_scaler(scaler_name)

            try:
                # Handle single column
                col_data = X_df[[col]].values
                scaled_data = scaler.fit_transform(col_data)
                X_scaled[col] = scaled_data.flatten()

                # Store scaler and info
                self.scalers[col] = scaler
                self.scaling_info[col] = {
                    "method": scaler_name,
                    "original_mean": X_df[col].mean(),
                    "original_std": X_df[col].std(),
                    "original_min": X_df[col].min(),
                    "original_max": X_df[col].max(),
                    "has_outliers": self._has_outliers(X_df[col]),
                }

                logger.info(f"Scaled column '{col}' using {scaler_name}")

            except Exception as e:
                logger.warning(f"Failed to scale column '{col}': {str(e)}")
                # Keep original values if scaling fails

        # Return in original format
        if isinstance(X, np.ndarray):
            return X_scaled.values
        return X_scaled

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform data using fitted scalers.

        Args:
            X: Input data

        Returns:
            Scaled data
        """
        if not self.scalers:
            raise ValueError("Scalers not fitted. Call fit_transform first.")

        X_df = self._ensure_dataframe(X)
        X_scaled = X_df.copy()

        for col, scaler in self.scalers.items():
            if col in X_df.columns:
                try:
                    col_data = X_df[[col]].values
                    scaled_data = scaler.transform(col_data)
                    X_scaled[col] = scaled_data.flatten()
                except Exception as e:
                    logger.warning(f"Failed to transform column '{col}': {str(e)}")

        # Return in original format
        if isinstance(X, np.ndarray):
            return X_scaled.values
        return X_scaled

    def inverse_transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Inverse transform scaled data back to original scale.

        Args:
            X: Scaled data

        Returns:
            Original scale data
        """
        if not self.scalers:
            raise ValueError("Scalers not fitted. Call fit_transform first.")

        X_df = self._ensure_dataframe(X)
        X_original = X_df.copy()

        for col, scaler in self.scalers.items():
            if col in X_df.columns:
                try:
                    col_data = X_df[[col]].values
                    original_data = scaler.inverse_transform(col_data)
                    X_original[col] = original_data.flatten()
                except Exception as e:
                    logger.warning(
                        f"Failed to inverse transform column '{col}': {str(e)}"
                    )

        # Return in original format
        if isinstance(X, np.ndarray):
            return X_original.values
        return X_original

    def _analyze_features(self, X_df: pd.DataFrame, exclude_columns: List[str]):
        """Analyze feature distributions for automatic scaling selection."""
        for col in X_df.columns:
            if col in exclude_columns:
                continue

            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(X_df[col]):
                continue

            # Analyze distribution
            col_data = X_df[col].dropna()

            if len(col_data) == 0:
                continue

            self.feature_distributions[col] = {
                "skewness": stats.skew(col_data),
                "kurtosis": stats.kurtosis(col_data),
                "has_outliers": self._has_outliers(col_data),
                "has_negative": (col_data < 0).any(),
                "has_zero": (col_data == 0).any(),
                "unique_ratio": len(col_data.unique()) / len(col_data),
                "distribution_test": self._test_distribution(col_data),
            }

    def _select_scaler_for_feature(self, feature_name: str) -> str:
        """Select appropriate scaler based on feature characteristics."""
        if feature_name not in self.feature_distributions:
            return "standard"

        dist_info = self.feature_distributions[feature_name]

        # Decision logic for scaler selection
        if dist_info["has_outliers"] and self.handle_outliers:
            # Use robust scaler for features with outliers
            return "robust"

        elif dist_info["unique_ratio"] < 0.05:
            # Discrete or categorical-like numeric features
            return "minmax"

        elif abs(dist_info["skewness"]) > 2:
            # Highly skewed distributions
            if dist_info["has_negative"]:
                return "power_yeo"  # Yeo-Johnson handles negative values
            elif not dist_info["has_zero"]:
                return "power_box"  # Box-Cox for positive values
            else:
                return "quantile_normal"

        elif dist_info["distribution_test"] == "uniform":
            # Uniform-like distributions
            return "minmax"

        elif dist_info["distribution_test"] == "normal":
            # Normal-like distributions
            return "standard"

        else:
            # Default to standard scaling
            return "standard"

    def _has_outliers(self, data: pd.Series, threshold: float = 3.0) -> bool:
        """Check if data has outliers using IQR method."""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        return ((data < lower_bound) | (data > upper_bound)).any()

    def _test_distribution(self, data: pd.Series) -> str:
        """Test distribution type of the data."""
        # Remove outliers for distribution testing
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        mask = (data >= Q1 - 1.5 * IQR) & (data <= Q3 + 1.5 * IQR)
        clean_data = data[mask]

        if len(clean_data) < 20:
            return "unknown"

        # Test for normality
        _, p_normal = stats.normaltest(clean_data)
        if p_normal > 0.05:
            return "normal"

        # Test for uniformity
        _, p_uniform = stats.kstest(
            clean_data,
            "uniform",
            args=(clean_data.min(), clean_data.max() - clean_data.min()),
        )
        if p_uniform > 0.05:
            return "uniform"

        return "other"

    def _create_scaler(self, scaler_name: str):
        """Create scaler instance."""
        if scaler_name not in self.available_scalers:
            logger.warning(f"Unknown scaler '{scaler_name}', using StandardScaler")
            return StandardScaler()

        scaler_class = self.available_scalers[scaler_name]
        return scaler_class() if callable(scaler_class) else scaler_class

    def _ensure_dataframe(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Ensure input is a DataFrame."""
        if isinstance(X, pd.DataFrame):
            return X
        else:
            return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

    def get_scaling_report(self) -> pd.DataFrame:
        """Get detailed report of scaling applied to each feature."""
        if not self.scaling_info:
            return pd.DataFrame()

        report_data = []
        for feature, info in self.scaling_info.items():
            report_data.append(
                {
                    "feature": feature,
                    "scaling_method": info["method"],
                    "original_mean": info["original_mean"],
                    "original_std": info["original_std"],
                    "original_range": f"[{info['original_min']:.2f}, {info['original_max']:.2f}]",
                    "has_outliers": info["has_outliers"],
                }
            )

        return pd.DataFrame(report_data)

    def recommend_scaling(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        task_type: str = "classification",
        model_type: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Recommend scaling strategy based on data and model type.

        Args:
            X: Input data
            task_type: "classification" or "regression"
            model_type: Type of model (e.g., "tree", "linear", "neural")

        Returns:
            Dictionary with recommendations
        """
        X_df = self._ensure_dataframe(X)
        recommendations = {}

        # Analyze all features
        self._analyze_features(X_df, [])

        # General recommendations
        if model_type == "tree":
            # Tree-based models don't require scaling
            recommendations["global"] = "none"
            recommendations["reason"] = "Tree-based models are scale-invariant"

        elif model_type == "linear":
            # Linear models benefit from standardization
            has_outliers = any(
                self.feature_distributions.get(col, {}).get("has_outliers", False)
                for col in X_df.columns
            )

            if has_outliers:
                recommendations["global"] = "robust"
                recommendations["reason"] = "Linear model with outliers present"
            else:
                recommendations["global"] = "standard"
                recommendations["reason"] = "Linear model without significant outliers"

        elif model_type == "neural":
            # Neural networks often work well with MinMax scaling
            recommendations["global"] = "minmax"
            recommendations["reason"] = (
                "Neural networks often perform better with [0,1] range"
            )

        else:
            # Default auto strategy
            recommendations["global"] = "auto"
            recommendations["reason"] = "Feature-specific scaling based on distribution"

        # Feature-specific recommendations
        feature_recommendations = {}
        for col in X_df.columns:
            if col in self.feature_distributions:
                feature_recommendations[col] = self._select_scaler_for_feature(col)

        recommendations["features"] = feature_recommendations

        return recommendations


class MultiScaler(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible multi-strategy scaler.

    Applies different scaling strategies to different feature groups.
    """

    def __init__(
        self,
        scaling_map: Optional[Dict[str, List[str]]] = None,
        default_scaler: str = "standard",
    ):
        """
        Initialize multi-scaler.

        Args:
            scaling_map: Dict mapping scaler names to feature lists
            default_scaler: Default scaler for unmapped features
        """
        self.scaling_map = scaling_map or {}
        self.default_scaler = default_scaler
        self.scalers_ = {}
        self.feature_names_ = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """Fit scalers to data."""
        X_df = self._ensure_dataframe(X)
        self.feature_names_ = X_df.columns.tolist()

        # Create reverse mapping
        feature_to_scaler = {}
        for scaler_name, features in self.scaling_map.items():
            for feature in features:
                if feature in self.feature_names_:
                    feature_to_scaler[feature] = scaler_name

        # Group features by scaler
        scaler_groups = {}
        for feature in self.feature_names_:
            scaler_name = feature_to_scaler.get(feature, self.default_scaler)
            if scaler_name not in scaler_groups:
                scaler_groups[scaler_name] = []
            scaler_groups[scaler_name].append(feature)

        # Fit scalers
        for scaler_name, features in scaler_groups.items():
            if scaler_name != "none":
                scaler = self._create_scaler(scaler_name)
                scaler.fit(X_df[features])
                self.scalers_[scaler_name] = {"scaler": scaler, "features": features}

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transform data using fitted scalers."""
        X_df = self._ensure_dataframe(X)
        X_transformed = X_df.copy()

        for scaler_name, scaler_info in self.scalers_.items():
            scaler = scaler_info["scaler"]
            features = scaler_info["features"]

            # Only transform features that exist in the input
            features_to_transform = [f for f in features if f in X_df.columns]
            if features_to_transform:
                X_transformed[features_to_transform] = scaler.transform(
                    X_df[features_to_transform]
                )

        return X_transformed.values

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y=None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Inverse transform data."""
        X_df = self._ensure_dataframe(X)
        X_original = X_df.copy()

        for scaler_name, scaler_info in self.scalers_.items():
            scaler = scaler_info["scaler"]
            features = scaler_info["features"]

            features_to_transform = [f for f in features if f in X_df.columns]
            if features_to_transform:
                X_original[features_to_transform] = scaler.inverse_transform(
                    X_df[features_to_transform]
                )

        return X_original.values

    def _ensure_dataframe(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Ensure input is a DataFrame."""
        if isinstance(X, pd.DataFrame):
            return X
        else:
            if self.feature_names_ is None:
                columns = [f"feature_{i}" for i in range(X.shape[1])]
            else:
                columns = self.feature_names_
            return pd.DataFrame(X, columns=columns)

    def _create_scaler(self, scaler_name: str):
        """Create scaler instance."""
        scaler_map = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
            "maxabs": MaxAbsScaler(),
            "quantile": QuantileTransformer(),
            "power": PowerTransformer(),
        }

        return scaler_map.get(scaler_name, StandardScaler())


class OutlierRobustScaler(BaseEstimator, TransformerMixin):
    """
    Custom outlier-robust scaler using winsorization.
    """

    def __init__(self, limits: Tuple[float, float] = (0.05, 0.95)):
        """
        Initialize outlier-robust scaler.

        Args:
            limits: Quantile limits for winsorization
        """
        self.limits = limits
        self.quantiles_ = {}
        self.scaler_ = None

    def fit(self, X: np.ndarray, y=None):
        """Fit the scaler."""
        X = np.asarray(X)

        # Calculate quantiles for each feature
        for i in range(X.shape[1]):
            lower_q = np.quantile(X[:, i], self.limits[0])
            upper_q = np.quantile(X[:, i], self.limits[1])
            self.quantiles_[i] = (lower_q, upper_q)

        # Winsorize and fit standard scaler
        X_winsorized = self._winsorize(X)
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X_winsorized)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data."""
        X = np.asarray(X)
        X_winsorized = self._winsorize(X)
        return self.scaler_.transform(X_winsorized)

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def _winsorize(self, X: np.ndarray) -> np.ndarray:
        """Apply winsorization to limit extreme values."""
        X_winsorized = X.copy()

        for i in range(X.shape[1]):
            if i in self.quantiles_:
                lower_q, upper_q = self.quantiles_[i]
                X_winsorized[:, i] = np.clip(X_winsorized[:, i], lower_q, upper_q)

        return X_winsorized


# Convenience functions
def auto_scale(
    X: Union[pd.DataFrame, np.ndarray], feature_types: Optional[Dict[str, str]] = None
) -> Tuple[Union[pd.DataFrame, np.ndarray], AdvancedScaler]:
    """
    Automatically scale data using intelligent feature analysis.

    Args:
        X: Input data
        feature_types: Optional feature type hints

    Returns:
        Tuple of (scaled data, fitted scaler)
    """
    scaler = AdvancedScaler(strategy="auto")
    X_scaled = scaler.fit_transform(X, feature_types)
    return X_scaled, scaler


def scale_for_model(
    X: Union[pd.DataFrame, np.ndarray], model_type: str
) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[AdvancedScaler, None]]:
    """
    Scale data appropriately for specific model type.

    Args:
        X: Input data
        model_type: Type of model ("tree", "linear", "neural", etc.)

    Returns:
        Tuple of (scaled data, scaler) or (original data, None) for tree models
    """
    if model_type == "tree":
        # Tree-based models don't need scaling
        return X, None

    scaler = AdvancedScaler()
    recommendations = scaler.recommend_scaling(X, model_type=model_type)

    if recommendations["global"] == "none":
        return X, None

    scaler.strategy = recommendations["global"]
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


# Export public API
__all__ = [
    "AdvancedScaler",
    "MultiScaler",
    "OutlierRobustScaler",
    "auto_scale",
    "scale_for_model",
]
