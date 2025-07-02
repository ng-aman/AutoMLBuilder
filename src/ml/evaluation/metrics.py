"""
Machine learning evaluation metrics for AutoML Builder.

This module provides comprehensive metrics for evaluating classification,
regression, and clustering models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn import metrics
from sklearn.metrics import (
    # Classification metrics
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    log_loss,
    matthews_corrcoef,
    cohen_kappa_score,
    balanced_accuracy_score,
    # Regression metrics
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score,
    max_error,
    mean_squared_log_error,
    # Clustering metrics
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    # Model selection metrics
    make_scorer,
)
import warnings
from dataclasses import dataclass, asdict


@dataclass
class MetricResult:
    """Container for metric results."""

    name: str
    value: float
    display_name: Optional[str] = None
    higher_is_better: bool = True
    confidence_interval: Optional[Tuple[float, float]] = None
    metadata: Optional[Dict[str, Any]] = None


class MetricsCalculator:
    """
    Comprehensive metrics calculator for ML models.

    Handles classification, regression, and clustering metrics with
    proper error handling and multi-class support.
    """

    def __init__(self):
        """Initialize metrics calculator."""
        self.classification_metrics = {
            "accuracy": (accuracy_score, True),
            "precision": (precision_score, True),
            "recall": (recall_score, True),
            "f1": (f1_score, True),
            "roc_auc": (roc_auc_score, True),
            "log_loss": (log_loss, False),
            "mcc": (matthews_corrcoef, True),
            "cohen_kappa": (cohen_kappa_score, True),
            "balanced_accuracy": (balanced_accuracy_score, True),
        }

        self.regression_metrics = {
            "mse": (mean_squared_error, False),
            "rmse": (
                lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
                False,
            ),
            "mae": (mean_absolute_error, False),
            "r2": (r2_score, True),
            "mape": (self._safe_mape, False),
            "explained_variance": (explained_variance_score, True),
            "max_error": (max_error, False),
        }

        self.clustering_metrics = {
            "silhouette": (silhouette_score, True),
            "calinski_harabasz": (calinski_harabasz_score, True),
            "davies_bouldin": (davies_bouldin_score, False),
        }

    def calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        average: str = "weighted",
        labels: Optional[List] = None,
    ) -> Dict[str, MetricResult]:
        """
        Calculate classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            average: Averaging strategy for multi-class
            labels: List of label names

        Returns:
            Dictionary of metric results
        """
        results = {}

        # Basic metrics
        results["accuracy"] = MetricResult(
            name="accuracy",
            value=accuracy_score(y_true, y_pred),
            display_name="Accuracy",
            higher_is_better=True,
        )

        # Multi-class metrics with averaging
        for metric_name in ["precision", "recall", "f1"]:
            try:
                metric_func = self.classification_metrics[metric_name][0]
                value = metric_func(y_true, y_pred, average=average, zero_division=0)

                results[metric_name] = MetricResult(
                    name=metric_name,
                    value=value,
                    display_name=metric_name.capitalize(),
                    higher_is_better=True,
                    metadata={"average": average},
                )
            except Exception as e:
                warnings.warn(f"Could not calculate {metric_name}: {str(e)}")

        # Probability-based metrics
        if y_proba is not None:
            # ROC AUC
            try:
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    if y_proba.ndim > 1:
                        auc_value = roc_auc_score(y_true, y_proba[:, 1])
                    else:
                        auc_value = roc_auc_score(y_true, y_proba)
                else:
                    # Multi-class
                    auc_value = roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average=average
                    )

                results["roc_auc"] = MetricResult(
                    name="roc_auc",
                    value=auc_value,
                    display_name="ROC AUC",
                    higher_is_better=True,
                )
            except Exception as e:
                warnings.warn(f"Could not calculate ROC AUC: {str(e)}")

            # Log loss
            try:
                loss_value = log_loss(y_true, y_proba)
                results["log_loss"] = MetricResult(
                    name="log_loss",
                    value=loss_value,
                    display_name="Log Loss",
                    higher_is_better=False,
                )
            except Exception as e:
                warnings.warn(f"Could not calculate log loss: {str(e)}")

        # Additional metrics
        try:
            results["mcc"] = MetricResult(
                name="mcc",
                value=matthews_corrcoef(y_true, y_pred),
                display_name="Matthews Correlation",
                higher_is_better=True,
            )
        except Exception:
            pass

        try:
            results["cohen_kappa"] = MetricResult(
                name="cohen_kappa",
                value=cohen_kappa_score(y_true, y_pred),
                display_name="Cohen's Kappa",
                higher_is_better=True,
            )
        except Exception:
            pass

        try:
            results["balanced_accuracy"] = MetricResult(
                name="balanced_accuracy",
                value=balanced_accuracy_score(y_true, y_pred),
                display_name="Balanced Accuracy",
                higher_is_better=True,
            )
        except Exception:
            pass

        # Per-class metrics
        if labels is not None:
            results["per_class_metrics"] = self._calculate_per_class_metrics(
                y_true, y_pred, labels
            )

        return results

    def calculate_regression_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, MetricResult]:
        """
        Calculate regression metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of metric results
        """
        results = {}

        # Mean Squared Error
        mse_value = mean_squared_error(y_true, y_pred)
        results["mse"] = MetricResult(
            name="mse",
            value=mse_value,
            display_name="Mean Squared Error",
            higher_is_better=False,
        )

        # Root Mean Squared Error
        results["rmse"] = MetricResult(
            name="rmse",
            value=np.sqrt(mse_value),
            display_name="Root Mean Squared Error",
            higher_is_better=False,
        )

        # Mean Absolute Error
        results["mae"] = MetricResult(
            name="mae",
            value=mean_absolute_error(y_true, y_pred),
            display_name="Mean Absolute Error",
            higher_is_better=False,
        )

        # R-squared
        results["r2"] = MetricResult(
            name="r2",
            value=r2_score(y_true, y_pred),
            display_name="R² Score",
            higher_is_better=True,
        )

        # Mean Absolute Percentage Error
        try:
            mape_value = self._safe_mape(y_true, y_pred)
            if mape_value is not None:
                results["mape"] = MetricResult(
                    name="mape",
                    value=mape_value,
                    display_name="Mean Absolute Percentage Error",
                    higher_is_better=False,
                )
        except Exception:
            pass

        # Explained Variance
        results["explained_variance"] = MetricResult(
            name="explained_variance",
            value=explained_variance_score(y_true, y_pred),
            display_name="Explained Variance",
            higher_is_better=True,
        )

        # Max Error
        results["max_error"] = MetricResult(
            name="max_error",
            value=max_error(y_true, y_pred),
            display_name="Maximum Error",
            higher_is_better=False,
        )

        # Additional statistics
        residuals = y_true - y_pred
        results["mean_residual"] = MetricResult(
            name="mean_residual",
            value=np.mean(residuals),
            display_name="Mean Residual",
            higher_is_better=False,
            metadata={"ideal_value": 0},
        )

        results["std_residual"] = MetricResult(
            name="std_residual",
            value=np.std(residuals),
            display_name="Std Dev of Residuals",
            higher_is_better=False,
        )

        return results

    def calculate_clustering_metrics(
        self, X: np.ndarray, labels: np.ndarray
    ) -> Dict[str, MetricResult]:
        """
        Calculate clustering metrics.

        Args:
            X: Feature matrix
            labels: Cluster labels

        Returns:
            Dictionary of metric results
        """
        results = {}

        # Check if we have more than one cluster
        n_clusters = len(np.unique(labels))
        if n_clusters <= 1:
            warnings.warn("Cannot calculate clustering metrics with only one cluster")
            return results

        # Silhouette Score
        try:
            results["silhouette"] = MetricResult(
                name="silhouette",
                value=silhouette_score(X, labels),
                display_name="Silhouette Score",
                higher_is_better=True,
                metadata={"range": [-1, 1]},
            )
        except Exception as e:
            warnings.warn(f"Could not calculate silhouette score: {str(e)}")

        # Calinski-Harabasz Score
        try:
            results["calinski_harabasz"] = MetricResult(
                name="calinski_harabasz",
                value=calinski_harabasz_score(X, labels),
                display_name="Calinski-Harabasz Score",
                higher_is_better=True,
            )
        except Exception as e:
            warnings.warn(f"Could not calculate Calinski-Harabasz score: {str(e)}")

        # Davies-Bouldin Score
        try:
            results["davies_bouldin"] = MetricResult(
                name="davies_bouldin",
                value=davies_bouldin_score(X, labels),
                display_name="Davies-Bouldin Score",
                higher_is_better=False,
                metadata={"ideal_value": 0},
            )
        except Exception as e:
            warnings.warn(f"Could not calculate Davies-Bouldin score: {str(e)}")

        # Cluster statistics
        cluster_counts = pd.Series(labels).value_counts()
        results["cluster_distribution"] = MetricResult(
            name="cluster_distribution",
            value=cluster_counts.std()
            / cluster_counts.mean(),  # Coefficient of variation
            display_name="Cluster Size Variation",
            higher_is_better=False,
            metadata={
                "cluster_sizes": cluster_counts.to_dict(),
                "n_clusters": n_clusters,
            },
        )

        return results

    def calculate_cv_metrics(
        self, cv_scores: List[Dict[str, float]]
    ) -> Dict[str, MetricResult]:
        """
        Calculate aggregated cross-validation metrics.

        Args:
            cv_scores: List of metric dictionaries from CV folds

        Returns:
            Aggregated metrics with confidence intervals
        """
        results = {}

        # Get all metric names
        all_metrics = set()
        for fold_scores in cv_scores:
            all_metrics.update(fold_scores.keys())

        # Calculate statistics for each metric
        for metric_name in all_metrics:
            values = [fold_scores.get(metric_name, np.nan) for fold_scores in cv_scores]
            values = [v for v in values if not np.isnan(v)]

            if values:
                mean_value = np.mean(values)
                std_value = np.std(values)

                # 95% confidence interval
                n = len(values)
                se = std_value / np.sqrt(n)
                ci_lower = mean_value - 1.96 * se
                ci_upper = mean_value + 1.96 * se

                results[metric_name] = MetricResult(
                    name=metric_name,
                    value=mean_value,
                    display_name=f"{metric_name} (CV)",
                    confidence_interval=(ci_lower, ci_upper),
                    metadata={"std": std_value, "n_folds": n, "fold_values": values},
                )

        return results

    def get_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List] = None,
        normalize: Optional[str] = None,
    ) -> Tuple[np.ndarray, List]:
        """
        Calculate confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: List of label names
            normalize: Normalization mode ('true', 'pred', 'all', None)

        Returns:
            Tuple of (confusion matrix, label names)
        """
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)

        if labels is None:
            labels = np.unique(np.concatenate([y_true, y_pred]))

        return cm, labels

    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List] = None,
        output_format: str = "dict",
    ) -> Union[Dict, str]:
        """
        Generate classification report.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: List of label names
            output_format: 'dict' or 'string'

        Returns:
            Classification report as dictionary or string
        """
        if output_format == "dict":
            return classification_report(
                y_true, y_pred, target_names=labels, output_dict=True, zero_division=0
            )
        else:
            return classification_report(
                y_true, y_pred, target_names=labels, zero_division=0
            )

    def get_roc_curve(
        self, y_true: np.ndarray, y_score: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate ROC curve.

        Args:
            y_true: True binary labels
            y_score: Prediction scores

        Returns:
            Tuple of (false positive rate, true positive rate, thresholds)
        """
        return roc_curve(y_true, y_score)

    def get_precision_recall_curve(
        self, y_true: np.ndarray, y_score: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate precision-recall curve.

        Args:
            y_true: True binary labels
            y_score: Prediction scores

        Returns:
            Tuple of (precision, recall, thresholds)
        """
        return precision_recall_curve(y_true, y_score)

    def _calculate_per_class_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, labels: List
    ) -> Dict[str, Dict[str, float]]:
        """Calculate per-class metrics."""
        report = classification_report(
            y_true, y_pred, target_names=labels, output_dict=True, zero_division=0
        )

        # Remove aggregate metrics
        per_class = {
            k: v
            for k, v in report.items()
            if k not in ["accuracy", "macro avg", "weighted avg"]
        }

        return per_class

    def _safe_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
        """Calculate MAPE with zero handling."""
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return None

        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def create_scorer(
        self, metric_name: str, problem_type: str = "classification", **kwargs
    ) -> Any:
        """
        Create a scikit-learn scorer for a metric.

        Args:
            metric_name: Name of the metric
            problem_type: Type of problem ('classification' or 'regression')
            **kwargs: Additional arguments for the scorer

        Returns:
            Scikit-learn scorer object
        """
        if problem_type == "classification":
            metric_dict = self.classification_metrics
        elif problem_type == "regression":
            metric_dict = self.regression_metrics
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")

        if metric_name not in metric_dict:
            raise ValueError(f"Unknown metric: {metric_name}")

        metric_func, higher_is_better = metric_dict[metric_name]

        return make_scorer(metric_func, greater_is_better=higher_is_better, **kwargs)

    def format_metrics(
        self, metrics: Dict[str, MetricResult], format_type: str = "table"
    ) -> Union[pd.DataFrame, str, Dict]:
        """
        Format metrics for display.

        Args:
            metrics: Dictionary of metric results
            format_type: Output format ('table', 'text', 'dict')

        Returns:
            Formatted metrics
        """
        if format_type == "table":
            data = []
            for metric_name, result in metrics.items():
                if metric_name == "per_class_metrics":
                    continue

                row = {
                    "Metric": result.display_name or result.name,
                    "Value": f"{result.value:.4f}",
                }

                if result.confidence_interval:
                    row["95% CI"] = (
                        f"[{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]"
                    )

                data.append(row)

            return pd.DataFrame(data)

        elif format_type == "text":
            lines = []
            for metric_name, result in metrics.items():
                if metric_name == "per_class_metrics":
                    continue

                name = result.display_name or result.name
                line = f"{name}: {result.value:.4f}"

                if result.confidence_interval:
                    line += f" (95% CI: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}])"

                lines.append(line)

            return "\n".join(lines)

        else:  # dict
            return {name: asdict(result) for name, result in metrics.items()}


# Convenience functions
def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    average: str = "weighted",
) -> Dict[str, float]:
    """
    Quick evaluation of classification model.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities
        average: Averaging strategy

    Returns:
        Dictionary of metric values
    """
    calculator = MetricsCalculator()
    results = calculator.calculate_classification_metrics(
        y_true, y_pred, y_proba, average
    )

    return {name: result.value for name, result in results.items()}


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Quick evaluation of regression model.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary of metric values
    """
    calculator = MetricsCalculator()
    results = calculator.calculate_regression_metrics(y_true, y_pred)

    return {name: result.value for name, result in results.items()}


def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Quick evaluation of clustering model.

    Args:
        X: Feature matrix
        labels: Cluster labels

    Returns:
        Dictionary of metric values
    """
    calculator = MetricsCalculator()
    results = calculator.calculate_clustering_metrics(X, labels)

    return {name: result.value for name, result in results.items()}


# Export public API
__all__ = [
    "MetricResult",
    "MetricsCalculator",
    "evaluate_classification",
    "evaluate_regression",
    "evaluate_clustering",
]
