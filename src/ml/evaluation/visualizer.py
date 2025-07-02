"""
Machine learning evaluation visualizations for AutoML Builder.

This module provides comprehensive visualization capabilities for model evaluation,
including confusion matrices, ROC curves, feature importance plots, and more.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import warnings


class ModelVisualizer:
    """
    Comprehensive visualizer for ML model evaluation.

    Creates interactive Plotly visualizations for various model types
    and evaluation metrics.
    """

    def __init__(self, style: str = "plotly_white"):
        """
        Initialize visualizer.

        Args:
            style: Plotly template style
        """
        self.style = style
        self.colors = px.colors.qualitative.Set3

        # Set default plot settings
        self.default_layout = {
            "template": style,
            "font": {"size": 12},
            "margin": {"l": 50, "r": 50, "t": 50, "b": 50},
            "showlegend": True,
            "hovermode": "closest",
        }

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        normalize: bool = True,
        title: str = "Confusion Matrix",
        colorscale: str = "Blues",
    ) -> go.Figure:
        """
        Create interactive confusion matrix heatmap.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            normalize: Whether to normalize values
            title: Plot title
            colorscale: Color scale for heatmap

        Returns:
            Plotly figure
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            value_format = ".2%"
        else:
            value_format = "d"

        # Create labels if not provided
        if labels is None:
            labels = [str(i) for i in range(cm.shape[0])]

        # Create annotation text
        annotations = []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                value = cm[i, j]
                if normalize:
                    text = f"{value:.1%}"
                else:
                    text = f"{int(value)}"

                annotations.append(
                    dict(
                        text=text,
                        x=labels[j],
                        y=labels[i],
                        showarrow=False,
                        font=dict(color="white" if value > cm.max() / 2 else "black"),
                    )
                )

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale=colorscale,
                hovertemplate="True: %{y}<br>Predicted: %{x}<br>Value: %{z}<extra></extra>",
            )
        )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            xaxis=dict(
                tickmode="array", tickvals=list(range(len(labels))), ticktext=labels
            ),
            yaxis=dict(
                tickmode="array", tickvals=list(range(len(labels))), ticktext=labels
            ),
            annotations=annotations,
            **self.default_layout,
        )

        return fig

    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_scores: Union[np.ndarray, Dict[str, np.ndarray]],
        labels: Optional[List[str]] = None,
        title: str = "ROC Curves",
        multi_class: str = "ovr",
    ) -> go.Figure:
        """
        Plot ROC curves for binary or multi-class classification.

        Args:
            y_true: True labels
            y_scores: Prediction scores (array or dict for multiple models)
            labels: Class labels
            title: Plot title
            multi_class: Strategy for multi-class ('ovr' or 'ovo')

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        # Handle multiple models
        if isinstance(y_scores, dict):
            for model_name, scores in y_scores.items():
                self._add_roc_curve(
                    fig, y_true, scores, model_name, labels, multi_class
                )
        else:
            self._add_roc_curve(fig, y_true, y_scores, "Model", labels, multi_class)

        # Add diagonal line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color="gray"),
                name="Random Classifier",
                showlegend=True,
            )
        )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0, 1.05]),
            yaxis=dict(range=[0, 1.05]),
            **self.default_layout,
        )

        return fig

    def plot_precision_recall_curves(
        self,
        y_true: np.ndarray,
        y_scores: Union[np.ndarray, Dict[str, np.ndarray]],
        labels: Optional[List[str]] = None,
        title: str = "Precision-Recall Curves",
    ) -> go.Figure:
        """
        Plot precision-recall curves.

        Args:
            y_true: True labels
            y_scores: Prediction scores
            labels: Class labels
            title: Plot title

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        # Handle multiple models
        if isinstance(y_scores, dict):
            for model_name, scores in y_scores.items():
                self._add_pr_curve(fig, y_true, scores, model_name)
        else:
            self._add_pr_curve(fig, y_true, y_scores, "Model")

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Recall",
            yaxis_title="Precision",
            xaxis=dict(range=[0, 1.05]),
            yaxis=dict(range=[0, 1.05]),
            **self.default_layout,
        )

        return fig

    def plot_feature_importance(
        self,
        importance_values: Union[np.ndarray, pd.Series, Dict[str, float]],
        feature_names: Optional[List[str]] = None,
        title: str = "Feature Importance",
        top_n: int = 20,
        orientation: str = "h",
    ) -> go.Figure:
        """
        Plot feature importance as bar chart.

        Args:
            importance_values: Feature importance values
            feature_names: Names of features
            title: Plot title
            top_n: Number of top features to show
            orientation: Bar orientation ('h' or 'v')

        Returns:
            Plotly figure
        """
        # Convert to DataFrame for easier handling
        if isinstance(importance_values, dict):
            df = pd.DataFrame(
                list(importance_values.items()), columns=["feature", "importance"]
            )
        elif isinstance(importance_values, pd.Series):
            df = pd.DataFrame(
                {
                    "feature": importance_values.index,
                    "importance": importance_values.values,
                }
            )
        else:
            if feature_names is None:
                feature_names = [f"Feature {i}" for i in range(len(importance_values))]
            df = pd.DataFrame(
                {"feature": feature_names, "importance": importance_values}
            )

        # Sort and take top N
        df = df.nlargest(top_n, "importance")

        # Create bar chart
        if orientation == "h":
            fig = px.bar(
                df,
                x="importance",
                y="feature",
                orientation="h",
                title=title,
                labels={"importance": "Importance", "feature": "Feature"},
                color="importance",
                color_continuous_scale="Viridis",
            )
            fig.update_yaxis(categoryorder="total ascending")
        else:
            fig = px.bar(
                df,
                x="feature",
                y="importance",
                orientation="v",
                title=title,
                labels={"importance": "Importance", "feature": "Feature"},
                color="importance",
                color_continuous_scale="Viridis",
            )
            fig.update_xaxis(tickangle=-45)

        fig.update_layout(**self.default_layout)

        return fig

    def plot_residuals(
        self, y_true: np.ndarray, y_pred: np.ndarray, title: str = "Residual Analysis"
    ) -> go.Figure:
        """
        Plot residual analysis for regression models.

        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title

        Returns:
            Plotly figure with subplots
        """
        residuals = y_true - y_pred

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Residuals vs Fitted",
                "Q-Q Plot",
                "Histogram of Residuals",
                "Residuals vs Index",
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
        )

        # 1. Residuals vs Fitted
        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=residuals,
                mode="markers",
                marker=dict(size=5, opacity=0.6),
                name="Residuals",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

        # 2. Q-Q Plot
        from scipy import stats

        theoretical_quantiles = stats.norm.ppf(
            (np.arange(len(residuals)) + 0.5) / len(residuals)
        )
        sample_quantiles = np.sort(residuals)

        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                mode="markers",
                marker=dict(size=5),
                name="Q-Q",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Add diagonal line
        min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
        max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(dash="dash", color="red"),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # 3. Histogram of residuals
        fig.add_trace(
            go.Histogram(x=residuals, nbinsx=30, name="Residuals", showlegend=False),
            row=2,
            col=1,
        )

        # 4. Residuals vs Index
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(residuals)),
                y=residuals,
                mode="markers",
                marker=dict(size=5, opacity=0.6),
                name="Residuals",
                showlegend=False,
            ),
            row=2,
            col=2,
        )

        # Update axes labels
        fig.update_xaxes(title_text="Fitted Values", row=1, col=1)
        fig.update_yaxes(title_text="Residuals", row=1, col=1)

        fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
        fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)

        fig.update_xaxes(title_text="Residuals", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)

        fig.update_xaxes(title_text="Index", row=2, col=2)
        fig.update_yaxes(title_text="Residuals", row=2, col=2)

        # Update layout
        fig.update_layout(title=title, height=800, **self.default_layout)

        return fig

    def plot_prediction_scatter(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Predictions vs Actual",
        show_metrics: bool = True,
    ) -> go.Figure:
        """
        Scatter plot of predictions vs actual values.

        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            show_metrics: Whether to show R² and RMSE

        Returns:
            Plotly figure
        """
        # Calculate metrics
        from sklearn.metrics import r2_score, mean_squared_error

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # Create scatter plot
        fig = go.Figure()

        # Add scatter points
        fig.add_trace(
            go.Scatter(
                x=y_true,
                y=y_pred,
                mode="markers",
                marker=dict(
                    size=8,
                    opacity=0.6,
                    color=y_true,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="True Value"),
                ),
                name="Predictions",
                hovertemplate="True: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>",
            )
        )

        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())

        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(dash="dash", color="red", width=2),
                name="Perfect Prediction",
                showlegend=True,
            )
        )

        # Update layout
        layout_update = {
            "title": title,
            "xaxis_title": "True Values",
            "yaxis_title": "Predicted Values",
            "xaxis": dict(range=[min_val * 0.95, max_val * 1.05]),
            "yaxis": dict(range=[min_val * 0.95, max_val * 1.05]),
        }

        if show_metrics:
            layout_update["title"] = (
                f"{title}<br><sub>R² = {r2:.3f}, RMSE = {rmse:.3f}</sub>"
            )

        fig.update_layout(**layout_update, **self.default_layout)

        return fig

    def plot_learning_curves(
        self,
        train_sizes: np.ndarray,
        train_scores: np.ndarray,
        val_scores: np.ndarray,
        title: str = "Learning Curves",
        metric_name: str = "Score",
    ) -> go.Figure:
        """
        Plot learning curves for model training.

        Args:
            train_sizes: Training set sizes
            train_scores: Training scores (shape: [n_sizes, n_cv_folds])
            val_scores: Validation scores (shape: [n_sizes, n_cv_folds])
            title: Plot title
            metric_name: Name of the metric

        Returns:
            Plotly figure
        """
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        fig = go.Figure()

        # Training scores
        fig.add_trace(
            go.Scatter(
                x=train_sizes,
                y=train_mean,
                mode="lines+markers",
                name="Training Score",
                line=dict(color="blue", width=2),
                marker=dict(size=8),
            )
        )

        # Training confidence interval
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([train_sizes, train_sizes[::-1]]),
                y=np.concatenate(
                    [train_mean + train_std, (train_mean - train_std)[::-1]]
                ),
                fill="toself",
                fillcolor="rgba(0,100,255,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
                name="Training CI",
            )
        )

        # Validation scores
        fig.add_trace(
            go.Scatter(
                x=train_sizes,
                y=val_mean,
                mode="lines+markers",
                name="Validation Score",
                line=dict(color="red", width=2),
                marker=dict(size=8),
            )
        )

        # Validation confidence interval
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([train_sizes, train_sizes[::-1]]),
                y=np.concatenate([val_mean + val_std, (val_mean - val_std)[::-1]]),
                fill="toself",
                fillcolor="rgba(255,100,0,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
                name="Validation CI",
            )
        )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Training Set Size",
            yaxis_title=metric_name,
            **self.default_layout,
        )

        return fig

    def plot_model_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        metric_names: Optional[List[str]] = None,
        title: str = "Model Comparison",
    ) -> go.Figure:
        """
        Compare multiple models across different metrics.

        Args:
            metrics_dict: Dict of {model_name: {metric_name: value}}
            metric_names: Metrics to include (None for all)
            title: Plot title

        Returns:
            Plotly figure
        """
        # Convert to DataFrame
        df = pd.DataFrame(metrics_dict).T

        if metric_names:
            df = df[metric_names]

        # Create grouped bar chart
        fig = go.Figure()

        for i, metric in enumerate(df.columns):
            fig.add_trace(
                go.Bar(
                    name=metric,
                    x=df.index,
                    y=df[metric],
                    marker_color=self.colors[i % len(self.colors)],
                )
            )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Model",
            yaxis_title="Score",
            barmode="group",
            **self.default_layout,
        )

        return fig

    def plot_classification_report_heatmap(
        self,
        classification_report: Dict[str, Dict[str, float]],
        title: str = "Classification Report Heatmap",
    ) -> go.Figure:
        """
        Create heatmap from classification report.

        Args:
            classification_report: Classification report dictionary
            title: Plot title

        Returns:
            Plotly figure
        """
        # Extract class metrics
        metrics = ["precision", "recall", "f1-score"]
        classes = [
            k
            for k in classification_report.keys()
            if k not in ["accuracy", "macro avg", "weighted avg"]
        ]

        # Create matrix
        matrix = []
        for cls in classes:
            row = [classification_report[cls].get(m, 0) for m in metrics]
            matrix.append(row)

        matrix = np.array(matrix)

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=metrics,
                y=classes,
                colorscale="RdYlBu",
                text=np.round(matrix, 3),
                texttemplate="%{text}",
                textfont={"size": 12},
                hovertemplate="Class: %{y}<br>Metric: %{x}<br>Value: %{z:.3f}<extra></extra>",
            )
        )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Metric",
            yaxis_title="Class",
            **self.default_layout,
        )

        return fig

    def _add_roc_curve(
        self,
        fig: go.Figure,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        name: str,
        labels: Optional[List[str]],
        multi_class: str,
    ):
        """Add ROC curve to figure."""
        n_classes = len(np.unique(y_true))

        if n_classes == 2:
            # Binary classification
            if y_scores.ndim > 1:
                y_scores = y_scores[:, 1]

            fpr, tpr, _ = roc_curve(y_true, y_scores)
            auc_score = auc(fpr, tpr)

            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{name} (AUC = {auc_score:.3f})",
                    hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
                )
            )
        else:
            # Multi-class classification
            y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
                auc_score = auc(fpr, tpr)

                class_name = labels[i] if labels else f"Class {i}"
                fig.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode="lines",
                        name=f"{name} - {class_name} (AUC = {auc_score:.3f})",
                    )
                )

    def _add_pr_curve(
        self, fig: go.Figure, y_true: np.ndarray, y_scores: np.ndarray, name: str
    ):
        """Add precision-recall curve to figure."""
        if y_scores.ndim > 1:
            y_scores = y_scores[:, 1]

        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        avg_precision = np.mean(precision)

        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=f"{name} (AP = {avg_precision:.3f})",
                hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
            )
        )


# Convenience functions
def plot_confusion_matrix(y_true, y_pred, **kwargs):
    """Quick confusion matrix plot."""
    visualizer = ModelVisualizer()
    return visualizer.plot_confusion_matrix(y_true, y_pred, **kwargs)


def plot_roc_curves(y_true, y_scores, **kwargs):
    """Quick ROC curves plot."""
    visualizer = ModelVisualizer()
    return visualizer.plot_roc_curves(y_true, y_scores, **kwargs)


def plot_feature_importance(importance_values, feature_names=None, **kwargs):
    """Quick feature importance plot."""
    visualizer = ModelVisualizer()
    return visualizer.plot_feature_importance(
        importance_values, feature_names, **kwargs
    )


def plot_residuals(y_true, y_pred, **kwargs):
    """Quick residual analysis plot."""
    visualizer = ModelVisualizer()
    return visualizer.plot_residuals(y_true, y_pred, **kwargs)


# Export public API
__all__ = [
    "ModelVisualizer",
    "plot_confusion_matrix",
    "plot_roc_curves",
    "plot_feature_importance",
    "plot_residuals",
]
