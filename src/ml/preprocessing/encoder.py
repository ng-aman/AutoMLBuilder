# src/ml/preprocessing/encoder.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.feature_extraction import FeatureHasher
from category_encoders import TargetEncoder, BinaryEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEncoder:
    """Handles categorical feature encoding"""

    def __init__(self):
        self.encoders = {}
        self.encoding_report = {
            "encoded_columns": {},
            "encoding_methods": {},
            "new_features_created": 0,
        }

    def encode(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        target_values: Optional[pd.Series] = None,
        strategy: str = "auto",
    ) -> pd.DataFrame:
        """
        Main encoding method

        Args:
            df: Input dataframe
            target_column: Target variable name
            target_values: Target values for target encoding
            strategy: Encoding strategy ('auto', 'onehot', 'label', 'target')

        Returns:
            Encoded dataframe
        """
        logger.info(f"Starting feature encoding with strategy: {strategy}")

        # Create a copy
        df = df.copy()

        # Get categorical columns
        categorical_columns = self._get_categorical_columns(df, target_column)

        if not categorical_columns:
            logger.info("No categorical columns to encode")
            return df

        # Apply encoding based on strategy
        if strategy == "auto":
            df = self._auto_encode(df, categorical_columns, target_values)
        elif strategy == "onehot":
            df = self._onehot_encode(df, categorical_columns)
        elif strategy == "label":
            df = self._label_encode(df, categorical_columns)
        elif strategy == "target":
            df = self._target_encode(df, categorical_columns, target_values)
        else:
            raise ValueError(f"Unknown encoding strategy: {strategy}")

        logger.info(
            f"Encoding complete. Created {self.encoding_report['new_features_created']} new features",
            report=self.encoding_report,
        )

        return df

    def _get_categorical_columns(
        self, df: pd.DataFrame, exclude: Optional[str] = None
    ) -> List[str]:
        """Get list of categorical columns"""
        categorical_columns = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        if exclude and exclude in categorical_columns:
            categorical_columns.remove(exclude)

        return categorical_columns

    def _auto_encode(
        self,
        df: pd.DataFrame,
        columns: List[str],
        target_values: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Automatically choose encoding method based on cardinality"""
        for column in columns:
            cardinality = df[column].nunique()

            if cardinality == 2:
                # Binary encoding
                df = self._binary_encode(df, [column])
            elif cardinality <= 10:
                # One-hot encoding for low cardinality
                df = self._onehot_encode(df, [column])
            elif cardinality <= 50 and target_values is not None:
                # Target encoding for medium cardinality
                df = self._target_encode(df, [column], target_values)
            else:
                # Label encoding for high cardinality
                df = self._label_encode(df, [column])

        return df

    def _onehot_encode(
        self, df: pd.DataFrame, columns: List[str], max_categories: int = 50
    ) -> pd.DataFrame:
        """Apply one-hot encoding"""
        for column in columns:
            # Check cardinality
            n_unique = df[column].nunique()

            if n_unique > max_categories:
                logger.warning(
                    f"Column '{column}' has {n_unique} categories. "
                    f"Skipping one-hot encoding."
                )
                continue

            # Get dummies
            dummies = pd.get_dummies(
                df[column],
                prefix=column,
                drop_first=True,  # Drop first to avoid multicollinearity
            )

            # Add to dataframe
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[column])

            # Update report
            self.encoding_report["encoded_columns"][column] = {
                "method": "onehot",
                "n_categories": n_unique,
                "n_features": len(dummies.columns),
            }
            self.encoding_report["new_features_created"] += len(dummies.columns)

            logger.info(
                f"One-hot encoded '{column}' -> {len(dummies.columns)} features"
            )

        return df

    def _label_encode(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Apply label encoding"""
        for column in columns:
            # Create and fit encoder
            encoder = LabelEncoder()

            # Handle unknown categories
            df[column] = df[column].fillna("unknown")
            df[column] = encoder.fit_transform(df[column])

            # Store encoder
            self.encoders[column] = encoder

            # Update report
            self.encoding_report["encoded_columns"][column] = {
                "method": "label",
                "n_categories": len(encoder.classes_),
            }

            logger.info(f"Label encoded '{column}'")

        return df

    def _target_encode(
        self,
        df: pd.DataFrame,
        columns: List[str],
        target_values: pd.Series,
        smoothing: float = 1.0,
    ) -> pd.DataFrame:
        """Apply target encoding"""
        if target_values is None:
            logger.warning("Target values required for target encoding")
            return self._label_encode(df, columns)

        for column in columns:
            # Create and fit encoder
            encoder = TargetEncoder(cols=[column], smoothing=smoothing)

            # Fit and transform
            df[column] = encoder.fit_transform(df[[column]], target_values)

            # Store encoder
            self.encoders[column] = encoder

            # Update report
            self.encoding_report["encoded_columns"][column] = {
                "method": "target",
                "smoothing": smoothing,
            }

            logger.info(f"Target encoded '{column}'")

        return df

    def _binary_encode(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Apply binary encoding for binary categorical variables"""
        for column in columns:
            # Get unique values
            unique_values = df[column].unique()

            if len(unique_values) == 2:
                # Map to 0 and 1
                mapping = {unique_values[0]: 0, unique_values[1]: 1}
                df[column] = df[column].map(mapping)

                # Update report
                self.encoding_report["encoded_columns"][column] = {
                    "method": "binary",
                    "mapping": mapping,
                }

                logger.info(f"Binary encoded '{column}'")
            else:
                logger.warning(
                    f"Column '{column}' has {len(unique_values)} unique values. "
                    f"Binary encoding requires exactly 2."
                )

        return df

    def transform(self, df: pd.DataFrame, fit_encoders: Dict[str, Any]) -> pd.DataFrame:
        """Transform new data using fitted encoders"""
        df = df.copy()

        for column, encoder in fit_encoders.items():
            if column in df.columns:
                if isinstance(encoder, LabelEncoder):
                    # Handle unknown categories
                    df[column] = df[column].fillna("unknown")
                    # Transform only known categories
                    known_categories = encoder.classes_
                    df[column] = df[column].apply(
                        lambda x: (
                            encoder.transform([x])[0] if x in known_categories else -1
                        )
                    )
                elif hasattr(encoder, "transform"):
                    df[column] = encoder.transform(df[[column]])

        return df

    def get_encoding_summary(self) -> Dict[str, Any]:
        """Get summary of encoding operations"""
        return self.encoding_report


class AdvancedEncoder(FeatureEncoder):
    """Advanced encoding techniques"""

    def frequency_encode(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Encode categories by their frequency"""
        for column in columns:
            # Calculate frequency
            frequency_map = df[column].value_counts().to_dict()

            # Apply mapping
            df[f"{column}_frequency"] = df[column].map(frequency_map)

            logger.info(f"Frequency encoded '{column}'")

        return df

    def hash_encode(
        self, df: pd.DataFrame, columns: List[str], n_features: int = 32
    ) -> pd.DataFrame:
        """Apply feature hashing (useful for high cardinality)"""
        hasher = FeatureHasher(n_features=n_features, input_type="string")

        for column in columns:
            # Convert to list of dicts
            features = df[column].apply(lambda x: {column: str(x)})

            # Apply hashing
            hashed = hasher.transform(features.tolist()).toarray()

            # Create column names
            hashed_columns = [f"{column}_hash_{i}" for i in range(n_features)]

            # Add to dataframe
            hashed_df = pd.DataFrame(hashed, columns=hashed_columns, index=df.index)
            df = pd.concat([df, hashed_df], axis=1)
            df = df.drop(columns=[column])

            logger.info(f"Hash encoded '{column}' -> {n_features} features")

        return df
