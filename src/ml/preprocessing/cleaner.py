# src/ml/preprocessing/cleaner.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from sklearn.impute import SimpleImputer, KNNImputer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataCleaner:
    """Handles data cleaning operations"""

    def __init__(self):
        self.cleaning_report = {
            "missing_values_handled": {},
            "duplicates_removed": 0,
            "outliers_handled": {},
            "columns_dropped": [],
            "rows_dropped": 0,
        }

    def clean(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        strategy: str = "auto",
    ) -> pd.DataFrame:
        """
        Main cleaning method

        Args:
            df: Input dataframe
            target_column: Target variable name
            strategy: Cleaning strategy ('auto', 'aggressive', 'conservative')

        Returns:
            Cleaned dataframe
        """
        logger.info(f"Starting data cleaning with strategy: {strategy}")
        original_shape = df.shape

        # Create a copy to avoid modifying original
        df = df.copy()

        # Handle duplicates
        df = self.handle_duplicates(df)

        # Handle missing values
        df = self.handle_missing_values(df, target_column, strategy)

        # Handle outliers
        df = self.handle_outliers(df, target_column, strategy)

        # Drop constant columns
        df = self.drop_constant_columns(df, target_column)

        # Drop high cardinality columns
        df = self.drop_high_cardinality_columns(df, target_column)

        logger.info(
            f"Cleaning complete. Shape: {original_shape} -> {df.shape}",
            report=self.cleaning_report,
        )

        return df

    def handle_duplicates(
        self, df: pd.DataFrame, subset: Optional[List[str]] = None, keep: str = "first"
    ) -> pd.DataFrame:
        """Remove duplicate rows"""
        initial_rows = len(df)
        df = df.drop_duplicates(subset=subset, keep=keep)

        duplicates_removed = initial_rows - len(df)
        self.cleaning_report["duplicates_removed"] = duplicates_removed

        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows")

        return df

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        strategy: str = "auto",
    ) -> pd.DataFrame:
        """Handle missing values"""
        missing_counts = df.isnull().sum()
        columns_with_missing = missing_counts[missing_counts > 0].to_dict()

        if not columns_with_missing:
            return df

        for column, missing_count in columns_with_missing.items():
            missing_ratio = missing_count / len(df)

            # Skip target column
            if column == target_column:
                continue

            # Strategy-based handling
            if strategy == "aggressive" or missing_ratio > 0.5:
                # Drop column if too many missing values
                df = df.drop(columns=[column])
                self.cleaning_report["columns_dropped"].append(column)
                logger.info(f"Dropped column '{column}' ({missing_ratio:.1%} missing)")

            elif missing_ratio > 0:
                # Impute missing values
                df = self._impute_column(df, column)
                self.cleaning_report["missing_values_handled"][column] = {
                    "count": missing_count,
                    "ratio": missing_ratio,
                    "method": "imputation",
                }

        return df

    def _impute_column(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Impute missing values in a column"""
        if df[column].dtype in ["object", "category"]:
            # Categorical: use mode
            imputer = SimpleImputer(strategy="most_frequent")
        else:
            # Numerical: use median
            imputer = SimpleImputer(strategy="median")

        df[column] = imputer.fit_transform(df[[column]])
        return df

    def handle_outliers(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        strategy: str = "auto",
    ) -> pd.DataFrame:
        """Handle outliers using IQR method"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        if target_column in numeric_columns:
            numeric_columns = numeric_columns.drop(target_column)

        for column in numeric_columns:
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1

            # Define bounds
            if strategy == "aggressive":
                multiplier = 1.5
            else:
                multiplier = 3.0

            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr

            # Count outliers
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

            if len(outliers) > 0:
                # Clip outliers
                df[column] = df[column].clip(lower_bound, upper_bound)

                self.cleaning_report["outliers_handled"][column] = {
                    "count": len(outliers),
                    "method": "clipping",
                    "bounds": (lower_bound, upper_bound),
                }

                logger.info(f"Clipped {len(outliers)} outliers in '{column}'")

        return df

    def drop_constant_columns(
        self, df: pd.DataFrame, target_column: Optional[str] = None
    ) -> pd.DataFrame:
        """Drop columns with constant values"""
        constant_columns = []

        for column in df.columns:
            if column == target_column:
                continue

            if df[column].nunique() <= 1:
                constant_columns.append(column)

        if constant_columns:
            df = df.drop(columns=constant_columns)
            self.cleaning_report["columns_dropped"].extend(constant_columns)
            logger.info(f"Dropped {len(constant_columns)} constant columns")

        return df

    def drop_high_cardinality_columns(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        threshold: float = 0.95,
    ) -> pd.DataFrame:
        """Drop columns with too many unique values"""
        high_cardinality_columns = []

        for column in df.columns:
            if column == target_column:
                continue

            if df[column].dtype in ["object", "category"]:
                cardinality_ratio = df[column].nunique() / len(df)

                if cardinality_ratio > threshold:
                    high_cardinality_columns.append(column)

        if high_cardinality_columns:
            df = df.drop(columns=high_cardinality_columns)
            self.cleaning_report["columns_dropped"].extend(high_cardinality_columns)
            logger.info(
                f"Dropped {len(high_cardinality_columns)} high cardinality columns"
            )

        return df

    def get_cleaning_summary(self) -> Dict[str, Any]:
        """Get summary of cleaning operations"""
        return self.cleaning_report


class AdvancedCleaner(DataCleaner):
    """Advanced data cleaning with more sophisticated methods"""

    def handle_missing_values_advanced(
        self, df: pd.DataFrame, target_column: Optional[str] = None, method: str = "knn"
    ) -> pd.DataFrame:
        """Advanced missing value imputation"""
        if method == "knn":
            # Separate numeric and categorical columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns

            if len(numeric_columns) > 0:
                # Use KNN imputer for numeric columns
                imputer = KNNImputer(n_neighbors=5)
                df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

        elif method == "iterative":
            # Use iterative imputer (experimental in sklearn)
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer

            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                imputer = IterativeImputer(random_state=42)
                df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

        return df

    def detect_and_handle_outliers_multivariate(
        self, df: pd.DataFrame, contamination: float = 0.1
    ) -> pd.DataFrame:
        """Detect outliers using multivariate methods"""
        from sklearn.ensemble import IsolationForest

        numeric_columns = df.select_dtypes(include=[np.number]).columns

        if len(numeric_columns) > 2:
            # Use Isolation Forest
            iso_forest = IsolationForest(contamination=contamination, random_state=42)

            outlier_labels = iso_forest.fit_predict(df[numeric_columns])

            # Mark outliers
            df["is_outlier"] = outlier_labels == -1

            logger.info(f"Detected {df['is_outlier'].sum()} multivariate outliers")

        return df
