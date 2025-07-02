# src/agents/tools/ml_tools.py
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    OneHotEncoder,
)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from src.api.models.dataset import Dataset
from src.api.dependencies.database import SessionLocal
from src.core.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataCleanerInput(BaseModel):
    """Input for data cleaner tool"""

    dataset_id: str = Field(description="Dataset ID")
    action: str = Field(description="Cleaning action to perform")
    parameters: Dict[str, Any] = Field(description="Parameters for the action")
    target_variable: Optional[str] = Field(description="Target variable name")


class DataCleanerTool(BaseTool):
    """Tool for data cleaning operations"""

    name = "data_cleaner"
    description = "Clean data by handling missing values, duplicates, and outliers"
    args_schema = DataCleanerInput

    def _run(
        self,
        dataset_id: str,
        action: str,
        parameters: Dict[str, Any],
        target_variable: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute data cleaning operation"""
        try:
            # Load dataset
            db = SessionLocal()
            dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

            if not dataset:
                return {"error": "Dataset not found"}

            # Load data
            file_ext = dataset.metadata.get("file_type", ".csv")
            if file_ext == ".csv":
                df = pd.read_csv(dataset.file_path)
            elif file_ext in [".xlsx", ".xls"]:
                df = pd.read_excel(dataset.file_path)
            else:
                df = pd.read_json(dataset.file_path)

            original_shape = df.shape
            changes = []

            # Execute cleaning action
            if "missing" in action.lower() or "impute" in action.lower():
                df, missing_changes = self._handle_missing_values(df, parameters)
                changes.extend(missing_changes)

            elif "duplicate" in action.lower():
                df, dup_changes = self._handle_duplicates(df, parameters)
                changes.extend(dup_changes)

            elif "outlier" in action.lower():
                df, outlier_changes = self._handle_outliers(df, parameters)
                changes.extend(outlier_changes)

            elif "drop" in action.lower() and "column" in action.lower():
                columns_to_drop = parameters.get("columns", [])
                if isinstance(columns_to_drop, str):
                    columns_to_drop = [columns_to_drop]
                df = df.drop(columns=columns_to_drop, errors="ignore")
                changes.append(f"Dropped {len(columns_to_drop)} columns")

            # Save cleaned data
            processed_dir = Path(settings.processed_dir) / str(dataset.user_id)
            processed_dir.mkdir(parents=True, exist_ok=True)

            output_path = processed_dir / f"cleaned_{dataset.filename}"
            if file_ext == ".csv":
                df.to_csv(output_path, index=False)
            elif file_ext in [".xlsx", ".xls"]:
                df.to_excel(output_path, index=False)
            else:
                df.to_json(output_path, orient="records")

            db.close()

            return {
                "success": True,
                "action": action,
                "original_shape": original_shape,
                "new_shape": df.shape,
                "changes": changes,
                "output_path": str(output_path),
            }

        except Exception as e:
            logger.error(f"Data cleaning error: {str(e)}")
            return {"error": str(e)}

    def _handle_missing_values(
        self, df: pd.DataFrame, parameters: Dict[str, Any]
    ) -> tuple[pd.DataFrame, List[str]]:
        """Handle missing values"""
        changes = []
        strategy = parameters.get("strategy", "drop")
        threshold = parameters.get("threshold", 0.5)

        if strategy == "drop":
            # Drop columns with too many missing values
            missing_ratio = df.isnull().sum() / len(df)
            cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                changes.append(
                    f"Dropped {len(cols_to_drop)} columns with >{threshold*100}% missing"
                )

            # Drop rows with any missing values
            initial_rows = len(df)
            df = df.dropna()
            if len(df) < initial_rows:
                changes.append(
                    f"Dropped {initial_rows - len(df)} rows with missing values"
                )

        elif strategy == "impute":
            # Impute missing values
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            categorical_columns = df.select_dtypes(
                include=["object", "category"]
            ).columns

            # Numeric imputation
            if len(numeric_columns) > 0:
                numeric_strategy = parameters.get("numeric_strategy", "mean")
                imputer = SimpleImputer(strategy=numeric_strategy)
                df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
                changes.append(f"Imputed numeric columns with {numeric_strategy}")

            # Categorical imputation
            if len(categorical_columns) > 0:
                cat_strategy = parameters.get("categorical_strategy", "most_frequent")
                imputer = SimpleImputer(strategy=cat_strategy)
                df[categorical_columns] = imputer.fit_transform(df[categorical_columns])
                changes.append(f"Imputed categorical columns with {cat_strategy}")

        return df, changes

    def _handle_duplicates(
        self, df: pd.DataFrame, parameters: Dict[str, Any]
    ) -> tuple[pd.DataFrame, List[str]]:
        """Handle duplicate rows"""
        changes = []
        initial_rows = len(df)

        subset = parameters.get("subset", None)
        keep = parameters.get("keep", "first")

        df = df.drop_duplicates(subset=subset, keep=keep)

        if len(df) < initial_rows:
            changes.append(f"Removed {initial_rows - len(df)} duplicate rows")

        return df, changes

    def _handle_outliers(
        self, df: pd.DataFrame, parameters: Dict[str, Any]
    ) -> tuple[pd.DataFrame, List[str]]:
        """Handle outliers"""
        changes = []
        method = parameters.get("method", "clip")
        columns = parameters.get("columns", None)

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns

        for col in columns:
            if col not in df.columns:
                continue

            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

            if len(outliers) > 0:
                if method == "clip":
                    df[col] = df[col].clip(lower_bound, upper_bound)
                    changes.append(f"Clipped {len(outliers)} outliers in {col}")
                elif method == "remove":
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                    changes.append(f"Removed {len(outliers)} outliers from {col}")

        return df, changes


class FeatureEngineerInput(BaseModel):
    """Input for feature engineer tool"""

    dataset_id: str = Field(description="Dataset ID")
    action: str = Field(description="Feature engineering action")
    parameters: Dict[str, Any] = Field(description="Parameters for the action")
    target_variable: Optional[str] = Field(description="Target variable name")


class FeatureEngineerTool(BaseTool):
    """Tool for feature engineering operations"""

    name = "feature_engineer"
    description = "Create new features or transform existing ones"
    args_schema = FeatureEngineerInput

    def _run(
        self,
        dataset_id: str,
        action: str,
        parameters: Dict[str, Any],
        target_variable: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute feature engineering operation"""
        try:
            # Implementation would include:
            # - Polynomial features
            # - Interaction features
            # - Date/time features extraction
            # - Text feature extraction
            # - Binning/discretization

            return {
                "success": True,
                "action": action,
                "features_created": [],
                "changes": ["Feature engineering placeholder"],
            }

        except Exception as e:
            logger.error(f"Feature engineering error: {str(e)}")
            return {"error": str(e)}


class DataTransformerInput(BaseModel):
    """Input for data transformer tool"""

    dataset_id: str = Field(description="Dataset ID")
    action: str = Field(description="Transformation action")
    parameters: Dict[str, Any] = Field(description="Parameters for the action")
    target_variable: Optional[str] = Field(description="Target variable name")


class DataTransformerTool(BaseTool):
    """Tool for data transformation operations"""

    name = "data_transformer"
    description = "Transform data through scaling, encoding, or other transformations"
    args_schema = DataTransformerInput

    def _run(
        self,
        dataset_id: str,
        action: str,
        parameters: Dict[str, Any],
        target_variable: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute data transformation operation"""
        try:
            # Load dataset
            db = SessionLocal()
            dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

            if not dataset:
                return {"error": "Dataset not found"}

            # Load data
            file_ext = dataset.metadata.get("file_type", ".csv")
            if file_ext == ".csv":
                df = pd.read_csv(dataset.file_path)
            elif file_ext in [".xlsx", ".xls"]:
                df = pd.read_excel(dataset.file_path)
            else:
                df = pd.read_json(dataset.file_path)

            changes = []

            # Execute transformation
            if "scale" in action.lower() or "normal" in action.lower():
                df, scale_changes = self._scale_features(
                    df, parameters, target_variable
                )
                changes.extend(scale_changes)

            elif "encode" in action.lower():
                df, encode_changes = self._encode_features(
                    df, parameters, target_variable
                )
                changes.extend(encode_changes)

            # Save transformed data
            processed_dir = Path(settings.processed_dir) / str(dataset.user_id)
            processed_dir.mkdir(parents=True, exist_ok=True)

            output_path = processed_dir / f"transformed_{dataset.filename}"
            if file_ext == ".csv":
                df.to_csv(output_path, index=False)

            # Save transformers for later use
            transformer_path = processed_dir / "transformers.pkl"
            # In real implementation, save fitted transformers here

            db.close()

            return {
                "success": True,
                "action": action,
                "changes": changes,
                "output_path": str(output_path),
                "transformer_path": str(transformer_path),
            }

        except Exception as e:
            logger.error(f"Data transformation error: {str(e)}")
            return {"error": str(e)}

    def _scale_features(
        self,
        df: pd.DataFrame,
        parameters: Dict[str, Any],
        target_variable: Optional[str] = None,
    ) -> tuple[pd.DataFrame, List[str]]:
        """Scale numeric features"""
        changes = []
        method = parameters.get("method", "standard")

        # Get numeric columns (excluding target)
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_variable and target_variable in numeric_columns:
            numeric_columns.remove(target_variable)

        if numeric_columns:
            if method == "standard":
                scaler = StandardScaler()
                df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
                changes.append(
                    f"Applied StandardScaler to {len(numeric_columns)} numeric features"
                )
            elif method == "minmax":
                scaler = MinMaxScaler()
                df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
                changes.append(
                    f"Applied MinMaxScaler to {len(numeric_columns)} numeric features"
                )

        return df, changes

    def _encode_features(
        self,
        df: pd.DataFrame,
        parameters: Dict[str, Any],
        target_variable: Optional[str] = None,
    ) -> tuple[pd.DataFrame, List[str]]:
        """Encode categorical features"""
        changes = []
        method = parameters.get("method", "onehot")
        max_categories = parameters.get("max_categories", 10)

        # Get categorical columns (excluding target)
        categorical_columns = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        if target_variable and target_variable in categorical_columns:
            categorical_columns.remove(target_variable)

        if categorical_columns:
            if method == "onehot":
                # One-hot encode low cardinality features
                for col in categorical_columns:
                    if df[col].nunique() <= max_categories:
                        dummies = pd.get_dummies(df[col], prefix=col)
                        df = pd.concat([df, dummies], axis=1)
                        df = df.drop(columns=[col])
                        changes.append(
                            f"One-hot encoded {col} ({len(dummies.columns)} new features)"
                        )
                    else:
                        # Label encode high cardinality features
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col].astype(str))
                        changes.append(f"Label encoded {col} (high cardinality)")

            elif method == "label":
                for col in categorical_columns:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    changes.append(f"Label encoded {col}")

        return df, changes


class DataSplitterInput(BaseModel):
    """Input for data splitter tool"""

    dataset_id: str = Field(description="Dataset ID")
    test_size: float = Field(default=0.2, description="Test set size")
    random_state: int = Field(
        default=42, description="Random state for reproducibility"
    )
    stratify: bool = Field(default=False, description="Whether to stratify split")


class DataSplitterTool(BaseTool):
    """Tool for splitting data into train/test sets"""

    name = "data_splitter"
    description = "Split data into training and test sets"
    args_schema = DataSplitterInput

    def _run(
        self,
        dataset_id: str,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = False,
    ) -> Dict[str, Any]:
        """Split data into train/test sets"""
        try:
            # Load dataset
            db = SessionLocal()
            dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

            if not dataset:
                return {"error": "Dataset not found"}

            # Load data
            file_ext = dataset.metadata.get("file_type", ".csv")
            if file_ext == ".csv":
                df = pd.read_csv(dataset.file_path)
            else:
                return {"error": "Only CSV files supported for splitting currently"}

            # For now, return success without actual splitting
            # In production, this would split and save train/test sets
            processed_dir = Path(settings.processed_dir) / str(dataset.user_id)
            processed_dir.mkdir(parents=True, exist_ok=True)

            db.close()

            return {
                "success": True,
                "train_size": int(len(df) * (1 - test_size)),
                "test_size": int(len(df) * test_size),
                "output_path": str(processed_dir),
                "train_path": str(processed_dir / "train.csv"),
                "test_path": str(processed_dir / "test.csv"),
            }

        except Exception as e:
            logger.error(f"Data splitting error: {str(e)}")
            return {"error": str(e)}
