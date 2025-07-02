# src/agents/tools/data_tools.py
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from src.api.models.dataset import Dataset
from src.api.dependencies.database import SessionLocal
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataProfilerInput(BaseModel):
    """Input for data profiler tool"""

    dataset_id: str = Field(description="ID of the dataset to profile")


class DataProfilerTool(BaseTool):
    """Tool for profiling datasets"""

    name = "data_profiler"
    description = "Profile a dataset to get basic statistics and information"
    args_schema = DataProfilerInput

    def _run(self, dataset_id: str) -> Dict[str, Any]:
        """Profile the dataset"""
        try:
            # Get dataset from database
            db = SessionLocal()
            dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

            if not dataset:
                return {"error": "Dataset not found"}

            # Load dataset
            file_ext = dataset.metadata.get("file_type", ".csv")
            if file_ext == ".csv":
                df = pd.read_csv(dataset.file_path)
            elif file_ext in [".xlsx", ".xls"]:
                df = pd.read_excel(dataset.file_path)
            elif file_ext == ".json":
                df = pd.read_json(dataset.file_path)
            else:
                return {"error": f"Unsupported file type: {file_ext}"}

            # Basic profiling
            profile = {
                "dataset_info": {
                    "dataset_id": dataset_id,
                    "filename": dataset.filename,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": df.columns.tolist(),
                    "column_types": df.dtypes.astype(str).to_dict(),
                    "memory_usage": df.memory_usage(deep=True).sum(),
                    "file_size": dataset.file_size,
                },
                "numeric_summary": {},
                "categorical_summary": {},
            }

            # Numeric columns summary
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                profile["numeric_summary"][col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "median": float(df[col].median()),
                    "q25": float(df[col].quantile(0.25)),
                    "q75": float(df[col].quantile(0.75)),
                    "missing": int(df[col].isna().sum()),
                    "unique": int(df[col].nunique()),
                }

            # Categorical columns summary
            categorical_columns = df.select_dtypes(
                include=["object", "category"]
            ).columns
            for col in categorical_columns:
                value_counts = df[col].value_counts()
                profile["categorical_summary"][col] = {
                    "unique": int(df[col].nunique()),
                    "missing": int(df[col].isna().sum()),
                    "most_frequent": (
                        str(value_counts.index[0]) if len(value_counts) > 0 else None
                    ),
                    "most_frequent_count": (
                        int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
                    ),
                    "top_5_values": value_counts.head(5).to_dict(),
                }

            db.close()
            return profile

        except Exception as e:
            logger.error(f"Data profiling error: {str(e)}")
            return {"error": str(e)}


class DataQualityInput(BaseModel):
    """Input for data quality tool"""

    dataset_info: Dict[str, Any] = Field(
        description="Dataset information from profiler"
    )


class DataQualityTool(BaseTool):
    """Tool for assessing data quality"""

    name = "data_quality"
    description = "Assess data quality issues like missing values, duplicates, outliers"
    args_schema = DataQualityInput

    def _run(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Assess data quality"""
        try:
            # Extract dataset ID and load data
            dataset_id = dataset_info.get("dataset_id")
            if not dataset_id:
                return {"error": "Dataset ID not provided"}

            db = SessionLocal()
            dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

            if not dataset:
                return {"error": "Dataset not found"}

            # Load dataset
            file_ext = dataset.metadata.get("file_type", ".csv")
            if file_ext == ".csv":
                df = pd.read_csv(dataset.file_path)
            elif file_ext in [".xlsx", ".xls"]:
                df = pd.read_excel(dataset.file_path)
            elif file_ext == ".json":
                df = pd.read_json(dataset.file_path)

            quality_report = {
                "missing_values": {},
                "duplicate_rows": 0,
                "duplicate_columns": [],
                "constant_columns": [],
                "outliers": {},
                "data_types_issues": [],
                "critical_issues": [],
            }

            # Missing values analysis
            missing_counts = df.isnull().sum()
            total_rows = len(df)
            for col, count in missing_counts.items():
                if count > 0:
                    percentage = (count / total_rows) * 100
                    quality_report["missing_values"][col] = percentage
                    if percentage > 90:
                        quality_report["critical_issues"].append(
                            f"Column '{col}' has {percentage:.1f}% missing values"
                        )

            # Duplicate rows
            quality_report["duplicate_rows"] = df.duplicated().sum()
            if quality_report["duplicate_rows"] > 0:
                dup_percentage = (quality_report["duplicate_rows"] / total_rows) * 100
                if dup_percentage > 10:
                    quality_report["critical_issues"].append(
                        f"{dup_percentage:.1f}% of rows are duplicates"
                    )

            # Constant columns
            for col in df.columns:
                if df[col].nunique() <= 1:
                    quality_report["constant_columns"].append(col)

            # Outliers detection (for numeric columns)
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                if len(outliers) > 0:
                    outlier_percentage = (len(outliers) / total_rows) * 100
                    quality_report["outliers"][col] = {
                        "count": len(outliers),
                        "percentage": outlier_percentage,
                        "lower_bound": float(lower_bound),
                        "upper_bound": float(upper_bound),
                    }

            # Data type issues
            for col in df.columns:
                # Check if numeric column has non-numeric values
                if col in numeric_columns:
                    try:
                        pd.to_numeric(df[col], errors="coerce")
                    except:
                        quality_report["data_types_issues"].append(
                            f"Column '{col}' expected to be numeric but contains non-numeric values"
                        )

            db.close()
            return quality_report

        except Exception as e:
            logger.error(f"Data quality assessment error: {str(e)}")
            return {"error": str(e)}


class TargetAnalyzerInput(BaseModel):
    """Input for target analyzer tool"""

    dataset_info: Dict[str, Any] = Field(description="Dataset information")
    target_variable: str = Field(description="Proposed target variable")


class TargetAnalyzerTool(BaseTool):
    """Tool for analyzing target variable"""

    name = "target_analyzer"
    description = "Analyze target variable to determine problem type"
    args_schema = TargetAnalyzerInput

    def _run(
        self, dataset_info: Dict[str, Any], target_variable: str
    ) -> Dict[str, Any]:
        """Analyze target variable"""
        try:
            # Validate target variable exists
            if target_variable not in dataset_info.get("column_names", []):
                return {
                    "error": f"Target variable '{target_variable}' not found in dataset",
                    "valid": False,
                }

            # Get dataset
            dataset_id = dataset_info.get("dataset_id")
            db = SessionLocal()
            dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

            # Load dataset
            file_ext = dataset.metadata.get("file_type", ".csv")
            if file_ext == ".csv":
                df = pd.read_csv(dataset.file_path)
            elif file_ext in [".xlsx", ".xls"]:
                df = pd.read_excel(dataset.file_path)
            elif file_ext == ".json":
                df = pd.read_json(dataset.file_path)

            target_series = df[target_variable]

            result = {
                "valid": True,
                "target_variable": target_variable,
                "problem_type": None,
                "target_info": {
                    "unique_values": int(target_series.nunique()),
                    "missing_values": int(target_series.isna().sum()),
                    "data_type": str(target_series.dtype),
                },
            }

            # Determine problem type
            if target_series.dtype in ["object", "category", "bool"]:
                result["problem_type"] = "classification"
                result["target_info"][
                    "class_distribution"
                ] = target_series.value_counts().to_dict()
                result["target_info"]["num_classes"] = len(
                    result["target_info"]["class_distribution"]
                )

                # Check for class imbalance
                class_counts = target_series.value_counts()
                if len(class_counts) > 1:
                    min_class = class_counts.min()
                    max_class = class_counts.max()
                    imbalance_ratio = max_class / min_class
                    result["target_info"]["imbalance_ratio"] = float(imbalance_ratio)
                    if imbalance_ratio > 10:
                        result["warnings"] = ["Severe class imbalance detected"]

            elif target_series.dtype in ["int64", "float64"]:
                # Check if it's actually categorical (low unique values)
                unique_ratio = target_series.nunique() / len(target_series)
                if unique_ratio < 0.05 and target_series.nunique() < 20:
                    result["problem_type"] = "classification"
                    result["target_info"]["likely_categorical"] = True
                else:
                    result["problem_type"] = "regression"
                    result["target_info"]["statistics"] = {
                        "mean": float(target_series.mean()),
                        "std": float(target_series.std()),
                        "min": float(target_series.min()),
                        "max": float(target_series.max()),
                        "skewness": float(target_series.skew()),
                        "kurtosis": float(target_series.kurtosis()),
                    }

            db.close()
            return result

        except Exception as e:
            logger.error(f"Target analysis error: {str(e)}")
            return {"error": str(e), "valid": False}


class FeatureAnalyzerInput(BaseModel):
    """Input for feature analyzer tool"""

    dataset_info: Dict[str, Any] = Field(description="Dataset information")
    target_variable: Optional[str] = Field(
        description="Target variable name", default=None
    )


class FeatureAnalyzerTool(BaseTool):
    """Tool for analyzing features"""

    name = "feature_analyzer"
    description = "Analyze feature characteristics and relationships"
    args_schema = FeatureAnalyzerInput

    def _run(
        self, dataset_info: Dict[str, Any], target_variable: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze features"""
        try:
            # Get dataset
            dataset_id = dataset_info.get("dataset_id")
            db = SessionLocal()
            dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

            # Load dataset
            file_ext = dataset.metadata.get("file_type", ".csv")
            if file_ext == ".csv":
                df = pd.read_csv(dataset.file_path)
            elif file_ext in [".xlsx", ".xls"]:
                df = pd.read_excel(dataset.file_path)
            elif file_ext == ".json":
                df = pd.read_json(dataset.file_path)

            # Separate features and target
            if target_variable and target_variable in df.columns:
                features = df.drop(columns=[target_variable])
                target = df[target_variable]
            else:
                features = df
                target = None

            analysis = {
                "numeric_features": [],
                "categorical_features": [],
                "datetime_features": [],
                "high_cardinality_categorical": [],
                "low_variance_features": [],
                "constant_features": [],
                "highly_correlated_features": [],
                "feature_importance": {},
            }

            # Classify features
            for col in features.columns:
                dtype = features[col].dtype
                unique_ratio = features[col].nunique() / len(features)

                if dtype in ["int64", "float64"]:
                    analysis["numeric_features"].append(col)

                    # Check for low variance
                    if features[col].std() < 0.01:
                        analysis["low_variance_features"].append(col)

                elif dtype in ["object", "category"]:
                    analysis["categorical_features"].append(col)

                    # Check cardinality
                    if features[col].nunique() > 50 or unique_ratio > 0.5:
                        analysis["high_cardinality_categorical"].append(col)

                elif "datetime" in str(dtype):
                    analysis["datetime_features"].append(col)

                # Check if constant
                if features[col].nunique() <= 1:
                    analysis["constant_features"].append(col)

            # Correlation analysis for numeric features
            if len(analysis["numeric_features"]) > 1:
                numeric_df = features[analysis["numeric_features"]]
                corr_matrix = numeric_df.corr()

                # Find highly correlated pairs
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > 0.95:
                            analysis["highly_correlated_features"].append(
                                {
                                    "feature1": corr_matrix.columns[i],
                                    "feature2": corr_matrix.columns[j],
                                    "correlation": float(corr_matrix.iloc[i, j]),
                                }
                            )

            # Check for varying scales
            if analysis["numeric_features"]:
                scales = {}
                for col in analysis["numeric_features"]:
                    scales[col] = features[col].max() - features[col].min()

                max_scale = max(scales.values())
                min_scale = min(scales.values())
                if max_scale / (min_scale + 1e-10) > 1000:
                    analysis["varying_scales"] = True

            db.close()
            return analysis

        except Exception as e:
            logger.error(f"Feature analysis error: {str(e)}")
            return {"error": str(e)}
