"""
Visualization tools for AutoML agents.
Provides various plotting functions for data analysis, model evaluation, and optimization results.
"""

import io
import base64
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import mlflow
import logging

logger = logging.getLogger(__name__)


class VisualizationTools:
    """Tools for creating various visualizations for AutoML pipeline."""

    def __init__(self, save_to_mlflow: bool = True):
        """
        Initialize visualization tools.

        Args:
            save_to_mlflow: Whether to save plots to MLflow
        """
        self.save_to_mlflow = save_to_mlflow

        # Set default themes
        plt.style.use("seaborn-v0_8-darkgrid")
        sns.set_palette("husl")

    def _save_figure(
        self, fig: Union[go.Figure, plt.Figure], name: str, plot_type: str = "plotly"
    ):
        """Save figure to MLflow if enabled."""
        if self.save_to_mlflow and mlflow.active_run():
            if plot_type == "plotly":
                mlflow.log_figure(fig, f"plots/{name}.html")
            else:  # matplotlib
                mlflow.log_figure(fig, f"plots/{name}.png")

    def plot_data_distribution(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Create distribution plots for numerical columns.

        Args:
            df: DataFrame to analyze
            columns: Specific columns to plot (if None, plots all numeric columns)

        Returns:
            Plotly figure with distribution plots
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        n_cols = min(len(columns), 4)
        n_rows = (len(columns) + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=columns[: len(columns)],
            vertical_spacing=0.1,
            horizontal_spacing=0.1,
        )

        for idx, col in enumerate(columns):
            row = idx // n_cols + 1
            col_idx = idx % n_cols + 1

            # Add histogram
            fig.add_trace(
                go.Histogram(
                    x=df[col].dropna(),
                    name=col,
                    nbinsx=30,
                    showlegend=False,
                    marker_color="rgba(0, 123, 255, 0.7)",
                ),
                row=row,
                col=col_idx,
            )

            # Add KDE line
            from scipy.stats import gaussian_kde

            values = df[col].dropna()
            if len(values) > 1:
                kde = gaussian_kde(values)
                x_range = np.linspace(values.min(), values.max(), 100)
                kde_values = kde(x_range)

                # Normalize KDE to match histogram scale
                hist, bins = np.histogram(values, bins=30)
                bin_width = bins[1] - bins[0]
                kde_values = kde_values * len(values) * bin_width

                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=kde_values,
                        mode="lines",
                        name="KDE",
                        showlegend=False,
                        line=dict(color="red", width=2),
                        yaxis=f"y{idx+1}",
                    ),
                    row=row,
                    col=col_idx,
                )

        fig.update_layout(
            title="Data Distribution Analysis", height=300 * n_rows, showlegend=False
        )

        self._save_figure(fig, "data_distribution")
        return fig

    def plot_correlation_matrix(
        self, df: pd.DataFrame, method: str = "pearson"
    ) -> go.Figure:
        """
        Create an interactive correlation matrix heatmap.

        Args:
            df: DataFrame to analyze
            method: Correlation method ('pearson', 'spearman', 'kendall')

        Returns:
            Plotly figure with correlation matrix
        """
        # Calculate correlation matrix
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr(method=method)

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale="RdBu",
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="Correlation"),
            )
        )

        fig.update_layout(
            title=f"{method.capitalize()} Correlation Matrix",
            xaxis=dict(tickangle=-45),
            width=800,
            height=800,
        )

        self._save_figure(fig, f"correlation_matrix_{method}")
        return fig

    def plot_missing_values(self, df: pd.DataFrame) -> go.Figure:
        """
        Create a visualization of missing values in the dataset.

        Args:
            df: DataFrame to analyze

        Returns:
            Plotly figure showing missing values
        """
        # Calculate missing values
        missing_counts = df.isnull().sum()
        missing_percent = (missing_counts / len(df)) * 100

        # Create DataFrame for plotting
        missing_df = pd.DataFrame(
            {
                "Column": missing_counts.index,
                "Missing Count": missing_counts.values,
                "Missing Percentage": missing_percent.values,
            }
        )
        missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values(
            "Missing Count", ascending=True
        )

        if len(missing_df) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No missing values found in the dataset!",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=20),
            )
        else:
            # Create horizontal bar chart
            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    y=missing_df["Column"],
                    x=missing_df["Missing Count"],
                    orientation="h",
                    text=[
                        f"{count} ({pct:.1f}%)"
                        for count, pct in zip(
                            missing_df["Missing Count"],
                            missing_df["Missing Percentage"],
                        )
                    ],
                    textposition="auto",
                    marker_color="indianred",
                )
            )

            fig.update_layout(
                title="Missing Values Analysis",
                xaxis_title="Number of Missing Values",
                yaxis_title="Columns",
                height=max(400, len(missing_df) * 30),
                showlegend=False,
            )

        self._save_figure(fig, "missing_values")
        return fig

    def plot_target_distribution(
        self, y: pd.Series, problem_type: str = "classification"
    ) -> go.Figure:
        """
        Plot distribution of target variable.

        Args:
            y: Target variable
            problem_type: 'classification' or 'regression'

        Returns:
            Plotly figure
        """
        if problem_type == "classification":
            # Count plot for classification
            value_counts = y.value_counts().sort_index()

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=value_counts.index.astype(str),
                        y=value_counts.values,
                        text=value_counts.values,
                        textposition="auto",
                        marker_color="lightblue",
                    )
                ]
            )

            fig.update_layout(
                title="Target Variable Distribution (Classification)",
                xaxis_title="Class",
                yaxis_title="Count",
                showlegend=False,
            )
        else:
            # Histogram for regression
            fig = go.Figure()

            fig.add_trace(
                go.Histogram(
                    x=y, nbinsx=50, marker_color="lightblue", name="Distribution"
                )
            )

            # Add statistics
            mean_val = y.mean()
            median_val = y.median()

            fig.add_vline(
                x=mean_val,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_val:.2f}",
            )
            fig.add_vline(
                x=median_val,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Median: {median_val:.2f}",
            )

            fig.update_layout(
                title="Target Variable Distribution (Regression)",
                xaxis_title="Value",
                yaxis_title="Count",
                showlegend=True,
            )

        self._save_figure(fig, "target_distribution")
        return fig

    def plot_feature_importance(
        self, feature_names: List[str], importance_values: np.ndarray, top_n: int = 20
    ) -> go.Figure:
        """
        Plot feature importance scores.

        Args:
            feature_names: List of feature names
            importance_values: Array of importance scores
            top_n: Number of top features to display

        Returns:
            Plotly figure
        """
        # Create DataFrame and sort by importance
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importance_values}
        ).sort_values("importance", ascending=True)

        # Select top N features
        if len(importance_df) > top_n:
            importance_df = importance_df.tail(top_n)

        # Create horizontal bar chart
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                y=importance_df["feature"],
                x=importance_df["importance"],
                orientation="h",
                marker_color="lightgreen",
                text=np.round(importance_df["importance"], 4),
                textposition="auto",
            )
        )

        fig.update_layout(
            title=f"Top {min(top_n, len(importance_df))} Feature Importances",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=max(400, len(importance_df) * 25),
            showlegend=False,
        )

        self._save_figure(fig, "feature_importance")
        return fig

    def plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Create an interactive confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels

        Returns:
            Plotly figure
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        if labels is None:
            labels = [str(i) for i in range(cm.shape[0])]

        # Create annotated heatmap
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=labels,
            y=labels,
            annotation_text=cm.astype(str),
            colorscale="Blues",
            showscale=True,
        )

        fig.update_layout(
            title="Confusion Matrix",
            xaxis=dict(title="Predicted Label", side="bottom"),
            yaxis=dict(title="True Label"),
            width=600,
            height=600,
        )

        # Fix axis labels
        fig["layout"]["xaxis"]["side"] = "bottom"

        self._save_figure(fig, "confusion_matrix")
        return fig

    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        class_names: Optional[List[str]] = None,
        multi_class: str = "ovr",
    ) -> go.Figure:
        """
        Plot ROC curves for binary or multiclass classification.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            class_names: Names of classes
            multi_class: 'ovr' (one-vs-rest) or 'ovo' (one-vs-one)

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        # Handle binary classification
        if y_proba.shape[1] == 2:
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)

            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"ROC curve (AUC = {roc_auc:.3f})",
                    line=dict(width=2),
                )
            )
        else:
            # Multiclass classification
            n_classes = y_proba.shape[1]

            if class_names is None:
                class_names = [f"Class {i}" for i in range(n_classes)]

            # Binarize the output
            y_true_bin = label_binarize(y_true, classes=range(n_classes))

            # Calculate ROC curve and AUC for each class
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)

                fig.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode="lines",
                        name=f"{class_names[i]} (AUC = {roc_auc:.3f})",
                        line=dict(width=2),
                    )
                )

        # Add diagonal line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Classifier",
                line=dict(color="gray", width=2, dash="dash"),
            )
        )

        fig.update_layout(
            title="ROC Curves",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0, 1.05]),
            yaxis=dict(range=[0, 1.05]),
            width=700,
            height=600,
            legend=dict(x=0.7, y=0.3),
        )

        self._save_figure(fig, "roc_curves")
        return fig

    def plot_precision_recall_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> go.Figure:
        """
        Plot precision-recall curves.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            class_names: Names of classes

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        # Handle binary classification
        if y_proba.shape[1] == 2:
            precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
            avg_precision = np.mean(precision)

            fig.add_trace(
                go.Scatter(
                    x=recall,
                    y=precision,
                    mode="lines",
                    name=f"PR curve (AP = {avg_precision:.3f})",
                    line=dict(width=2),
                )
            )
        else:
            # Multiclass classification
            n_classes = y_proba.shape[1]

            if class_names is None:
                class_names = [f"Class {i}" for i in range(n_classes)]

            # Binarize the output
            y_true_bin = label_binarize(y_true, classes=range(n_classes))

            # Calculate PR curve for each class
            for i in range(n_classes):
                precision, recall, _ = precision_recall_curve(
                    y_true_bin[:, i], y_proba[:, i]
                )
                avg_precision = np.mean(precision)

                fig.add_trace(
                    go.Scatter(
                        x=recall,
                        y=precision,
                        mode="lines",
                        name=f"{class_names[i]} (AP = {avg_precision:.3f})",
                        line=dict(width=2),
                    )
                )

        fig.update_layout(
            title="Precision-Recall Curves",
            xaxis_title="Recall",
            yaxis_title="Precision",
            xaxis=dict(range=[0, 1.05]),
            yaxis=dict(range=[0, 1.05]),
            width=700,
            height=600,
        )

        self._save_figure(fig, "precision_recall_curves")
        return fig

    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
        """
        Plot residuals for regression analysis.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Plotly figure with residual plots
        """
        residuals = y_true - y_pred

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Residuals vs Predicted",
                "Residual Distribution",
                "Q-Q Plot",
                "Scale-Location",
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
        )

        # 1. Residuals vs Predicted
        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=residuals,
                mode="markers",
                marker=dict(size=5, opacity=0.6),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

        # 2. Residual Distribution
        fig.add_trace(
            go.Histogram(x=residuals, nbinsx=30, showlegend=False), row=1, col=2
        )

        # 3. Q-Q Plot
        from scipy import stats

        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
        sample_quantiles = np.sort(residuals)

        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                mode="markers",
                marker=dict(size=5),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Add diagonal line for Q-Q plot
        min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
        max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color="red", dash="dash"),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # 4. Scale-Location Plot
        standardized_residuals = residuals / np.std(residuals)
        sqrt_abs_residuals = np.sqrt(np.abs(standardized_residuals))

        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=sqrt_abs_residuals,
                mode="markers",
                marker=dict(size=5, opacity=0.6),
                showlegend=False,
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_xaxes(title_text="Predicted Values", row=1, col=1)
        fig.update_yaxes(title_text="Residuals", row=1, col=1)

        fig.update_xaxes(title_text="Residuals", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)

        fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=1)
        fig.update_yaxes(title_text="Sample Quantiles", row=2, col=1)

        fig.update_xaxes(title_text="Predicted Values", row=2, col=2)
        fig.update_yaxes(title_text="√|Standardized Residuals|", row=2, col=2)

        fig.update_layout(title="Residual Analysis", height=800, showlegend=False)

        self._save_figure(fig, "residual_analysis")
        return fig

    def plot_learning_curves(
        self,
        train_sizes: List[int],
        train_scores: List[float],
        val_scores: List[float],
        metric_name: str = "Score",
    ) -> go.Figure:
        """
        Plot learning curves to diagnose overfitting/underfitting.

        Args:
            train_sizes: List of training set sizes
            train_scores: Training scores for each size
            val_scores: Validation scores for each size
            metric_name: Name of the metric being plotted

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        # Training scores
        fig.add_trace(
            go.Scatter(
                x=train_sizes,
                y=train_scores,
                mode="lines+markers",
                name="Training Score",
                line=dict(color="blue", width=2),
                marker=dict(size=8),
            )
        )

        # Validation scores
        fig.add_trace(
            go.Scatter(
                x=train_sizes,
                y=val_scores,
                mode="lines+markers",
                name="Validation Score",
                line=dict(color="red", width=2),
                marker=dict(size=8),
            )
        )

        fig.update_layout(
            title="Learning Curves",
            xaxis_title="Training Set Size",
            yaxis_title=metric_name,
            hovermode="x unified",
            width=700,
            height=500,
        )

        self._save_figure(fig, "learning_curves")
        return fig

    def plot_hyperparameter_optimization(self, study) -> go.Figure:
        """
        Plot Optuna hyperparameter optimization results.

        Args:
            study: Optuna study object

        Returns:
            Plotly figure with optimization visualization
        """
        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Optimization History",
                "Parallel Coordinates",
                "Parameter Importances",
                "Best Parameters",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "parcoords"}],
                [{"type": "bar"}, {"type": "table"}],
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
        )

        # 1. Optimization History
        trials = study.trials
        trial_numbers = [t.number for t in trials]
        values = [t.value for t in trials if t.value is not None]

        fig.add_trace(
            go.Scatter(
                x=trial_numbers[: len(values)],
                y=values,
                mode="lines+markers",
                name="Objective Value",
                line=dict(color="blue", width=2),
                marker=dict(size=6),
            ),
            row=1,
            col=1,
        )

        # Add best value line
        best_values = [min(values[: i + 1]) for i in range(len(values))]
        fig.add_trace(
            go.Scatter(
                x=trial_numbers[: len(values)],
                y=best_values,
                mode="lines",
                name="Best Value",
                line=dict(color="red", width=2, dash="dash"),
            ),
            row=1,
            col=1,
        )

        # 2. Parallel Coordinates (if multiple parameters)
        if len(study.best_params) > 1:
            param_names = list(study.best_params.keys())
            param_values = {
                name: [t.params.get(name, None) for t in trials] for name in param_names
            }

            dimensions = []
            for name in param_names:
                values = param_values[name]
                # Filter out None values
                valid_values = [v for v in values if v is not None]
                if valid_values:
                    dimensions.append(
                        dict(
                            label=name,
                            values=valid_values[: len(trial_numbers)],
                            range=[min(valid_values), max(valid_values)],
                        )
                    )

            if dimensions:
                fig.add_trace(
                    go.Parcoords(
                        dimensions=dimensions,
                        line=dict(
                            color=[t.value for t in trials if t.value is not None][
                                : len(trial_numbers)
                            ],
                            colorscale="Viridis",
                            showscale=True,
                        ),
                    ),
                    row=1,
                    col=2,
                )

        # 3. Parameter Importances (placeholder - would need actual importance calculation)
        # For now, show parameter ranges
        param_ranges = []
        param_names_list = []

        for name, value in study.best_params.items():
            param_names_list.append(name)
            # Calculate range from trials
            param_vals = [
                t.params.get(name, None)
                for t in trials
                if t.params.get(name) is not None
            ]
            if param_vals:
                param_range = max(param_vals) - min(param_vals)
                param_ranges.append(param_range)
            else:
                param_ranges.append(0)

        if param_ranges:
            fig.add_trace(
                go.Bar(
                    x=param_names_list,
                    y=param_ranges,
                    name="Parameter Range",
                    marker_color="lightgreen",
                ),
                row=2,
                col=1,
            )

        # 4. Best Parameters Table
        best_params_data = [[k, v] for k, v in study.best_params.items()]

        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Parameter", "Best Value"], align="left", font=dict(size=12)
                ),
                cells=dict(
                    values=(
                        list(zip(*best_params_data)) if best_params_data else [[], []]
                    ),
                    align="left",
                    font=dict(size=11),
                ),
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_xaxes(title_text="Trial Number", row=1, col=1)
        fig.update_yaxes(title_text="Objective Value", row=1, col=1)
        fig.update_xaxes(title_text="Parameter", row=2, col=1)
        fig.update_yaxes(title_text="Range", row=2, col=1)

        fig.update_layout(
            title=f"Hyperparameter Optimization Results (Best Value: {study.best_value:.4f})",
            height=800,
            showlegend=True,
        )

        self._save_figure(fig, "hyperparameter_optimization")
        return fig

    def create_model_comparison_report(
        self, results: Dict[str, Dict[str, float]]
    ) -> go.Figure:
        """
        Create a comprehensive model comparison report.

        Args:
            results: Dictionary mapping model names to their metrics
                    e.g., {'RandomForest': {'accuracy': 0.95, 'f1': 0.94}, ...}

        Returns:
            Plotly figure with model comparison
        """
        # Prepare data
        models = list(results.keys())
        metrics = list(next(iter(results.values())).keys())

        # Create subplots
        n_metrics = len(metrics)
        fig = make_subplots(
            rows=(n_metrics + 1) // 2,
            cols=2,
            subplot_titles=metrics,
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
        )

        # Plot each metric
        for idx, metric in enumerate(metrics):
            row = idx // 2 + 1
            col = idx % 2 + 1

            values = [results[model][metric] for model in models]

            fig.add_trace(
                go.Bar(
                    x=models,
                    y=values,
                    name=metric,
                    text=[f"{v:.3f}" for v in values],
                    textposition="auto",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

            fig.update_xaxes(tickangle=-45, row=row, col=col)
            fig.update_yaxes(title_text=metric, row=row, col=col)

        fig.update_layout(
            title="Model Performance Comparison",
            height=400 * ((n_metrics + 1) // 2),
            showlegend=False,
        )

        self._save_figure(fig, "model_comparison")
        return fig

    def plot_prediction_vs_actual(
        self, y_true: np.ndarray, y_pred: np.ndarray, sample_size: int = 1000
    ) -> go.Figure:
        """
        Create scatter plot of predictions vs actual values for regression.

        Args:
            y_true: True values
            y_pred: Predicted values
            sample_size: Number of points to plot (for performance)

        Returns:
            Plotly figure
        """
        # Sample data if too large
        if len(y_true) > sample_size:
            indices = np.random.choice(len(y_true), sample_size, replace=False)
            y_true_sample = y_true[indices]
            y_pred_sample = y_pred[indices]
        else:
            y_true_sample = y_true
            y_pred_sample = y_pred

        # Create scatter plot
        fig = go.Figure()

        # Add scatter points
        fig.add_trace(
            go.Scatter(
                x=y_true_sample,
                y=y_pred_sample,
                mode="markers",
                name="Predictions",
                marker=dict(
                    size=8,
                    opacity=0.6,
                    color=np.abs(y_true_sample - y_pred_sample),
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Absolute Error"),
                ),
            )
        )

        # Add perfect prediction line
        min_val = min(y_true_sample.min(), y_pred_sample.min())
        max_val = max(y_true_sample.max(), y_pred_sample.max())

        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Perfect Prediction",
                line=dict(color="red", width=2, dash="dash"),
            )
        )

        # Calculate R²
        from sklearn.metrics import r2_score

        r2 = r2_score(y_true, y_pred)

        fig.update_layout(
            title=f"Predictions vs Actual Values (R² = {r2:.3f})",
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values",
            width=700,
            height=700,
        )

        self._save_figure(fig, "prediction_vs_actual")
        return fig


# Convenience functions for direct usage by agents
def create_visualizer(save_to_mlflow: bool = True) -> VisualizationTools:
    """Create and return a VisualizationTools instance."""
    return VisualizationTools(save_to_mlflow=save_to_mlflow)


def plot_eda_report(
    df: pd.DataFrame, target_column: Optional[str] = None
) -> Dict[str, go.Figure]:
    """
    Generate a complete EDA report with multiple visualizations.

    Args:
        df: DataFrame to analyze
        target_column: Name of target column (if applicable)

    Returns:
        Dictionary of plot names to figures
    """
    viz = VisualizationTools()
    plots = {}

    # Data distribution
    plots["distribution"] = viz.plot_data_distribution(df)

    # Missing values
    plots["missing"] = viz.plot_missing_values(df)

    # Correlation matrix
    plots["correlation"] = viz.plot_correlation_matrix(df)

    # Target distribution if specified
    if target_column and target_column in df.columns:
        problem_type = (
            "classification" if df[target_column].nunique() < 20 else "regression"
        )
        plots["target"] = viz.plot_target_distribution(df[target_column], problem_type)

    return plots


def plot_model_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    problem_type: str = "classification",
) -> Dict[str, go.Figure]:
    """
    Generate a complete model evaluation report.

    Args:
        y_true: True labels/values
        y_pred: Predicted labels/values
        y_proba: Predicted probabilities (for classification)
        problem_type: 'classification' or 'regression'

    Returns:
        Dictionary of plot names to figures
    """
    viz = VisualizationTools()
    plots = {}

    if problem_type == "classification":
        # Confusion matrix
        plots["confusion_matrix"] = viz.plot_confusion_matrix(y_true, y_pred)

        # ROC curves (if probabilities available)
        if y_proba is not None:
            plots["roc_curves"] = viz.plot_roc_curves(y_true, y_proba)
            plots["pr_curves"] = viz.plot_precision_recall_curves(y_true, y_proba)
    else:
        # Regression plots
        plots["prediction_vs_actual"] = viz.plot_prediction_vs_actual(y_true, y_pred)
        plots["residuals"] = viz.plot_residuals(y_true, y_pred)

    return plots
