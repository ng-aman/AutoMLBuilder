# src/ml/optimization/optuna_optimizer.py
import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import make_scorer
import mlflow
import mlflow.sklearn
import warnings
import joblib
from datetime import datetime
from src.utils.logger import get_logger
from src.core.config import settings
from .search_spaces import SearchSpaceBuilder

logger = get_logger(__name__)


class OptunaOptimizer:
    """
    Hyperparameter optimizer using Optuna framework.

    Integrates with MLflow for experiment tracking and supports
    classification, regression, and custom objective functions.
    """

    def __init__(
        self,
        task_type: str = "classification",
        metric: str = None,
        direction: str = None,
        n_trials: int = None,
        timeout: int = None,
        n_jobs: int = 1,
        cv_folds: int = 5,
        random_state: int = 42,
        storage: Optional[str] = None,
        study_name: Optional[str] = None,
        mlflow_tracking: bool = True,
    ):
        """
        Initialize Optuna optimizer.

        Args:
            task_type: Type of ML task ("classification" or "regression")
            metric: Optimization metric (e.g., "accuracy", "r2", "f1")
            direction: "maximize" or "minimize"
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            n_jobs: Number of parallel jobs
            cv_folds: Number of cross-validation folds
            random_state: Random seed
            storage: Optuna storage backend URL
            study_name: Name for the study
            mlflow_tracking: Whether to track with MLflow
        """
        self.task_type = task_type
        self.metric = metric or ("accuracy" if task_type == "classification" else "r2")
        self.direction = direction or self._get_default_direction()
        self.n_trials = n_trials or settings.models.optuna_n_trials
        self.timeout = timeout or settings.models.optuna_timeout
        self.n_jobs = n_jobs or settings.models.optuna_n_jobs
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.storage = storage
        self.study_name = (
            study_name
            or f"{settings.models.optuna_study_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.mlflow_tracking = mlflow_tracking

        # Initialize components
        self.search_space_builder = SearchSpaceBuilder()
        self.study = None
        self.best_model = None
        self.best_params = None
        self.best_score = None
        self.optimization_history = []

        # Set up cross-validation
        if task_type == "classification":
            self.cv = StratifiedKFold(
                n_splits=cv_folds, shuffle=True, random_state=random_state
            )
        else:
            self.cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def _get_default_direction(self) -> str:
        """Get default optimization direction based on metric."""
        minimize_metrics = [
            "mse",
            "rmse",
            "mae",
            "log_loss",
            "neg_log_loss",
            "error",
            "max_error",
        ]

        if any(m in self.metric.lower() for m in minimize_metrics):
            return "minimize"
        return "maximize"

    def optimize(
        self,
        model_class: Any,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        search_space: Optional[Dict[str, Any]] = None,
        custom_objective: Optional[Callable] = None,
        callbacks: Optional[List[Callable]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a model.

        Args:
            model_class: Model class or string name
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            search_space: Custom search space (optional)
            custom_objective: Custom objective function (optional)
            callbacks: List of Optuna callbacks
            **kwargs: Additional arguments for the model

        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting Optuna optimization for {model_class}")
        logger.info(
            f"Metric: {self.metric}, Direction: {self.direction}, Trials: {self.n_trials}"
        )

        # Get model name
        if isinstance(model_class, str):
            model_name = model_class
            model_class = self._get_model_class(model_name)
        else:
            model_name = model_class.__name__

        # Get or create search space
        if search_space is None:
            search_space = self.search_space_builder.get_search_space(
                model_name, self.task_type
            )

        # Create objective function
        if custom_objective is None:
            objective = self._create_objective(
                model_class, X_train, y_train, X_val, y_val, search_space, **kwargs
            )
        else:
            objective = custom_objective

        # Create or load study
        sampler = optuna.samplers.TPESampler(seed=self.random_state)

        if self.storage:
            self.study = optuna.create_study(
                study_name=self.study_name,
                direction=self.direction,
                storage=self.storage,
                sampler=sampler,
                load_if_exists=True,
            )
        else:
            self.study = optuna.create_study(
                study_name=self.study_name, direction=self.direction, sampler=sampler
            )

        # Add callbacks
        all_callbacks = callbacks or []
        all_callbacks.extend(self._get_default_callbacks())

        # Start MLflow run if enabled
        if self.mlflow_tracking:
            mlflow.set_experiment(f"optuna_{model_name}")
            mlflow.start_run(run_name=f"optimization_{self.study_name}")
            mlflow.log_params(
                {
                    "model": model_name,
                    "metric": self.metric,
                    "direction": self.direction,
                    "n_trials": self.n_trials,
                    "cv_folds": self.cv_folds,
                }
            )

        try:
            # Run optimization
            self.study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                n_jobs=self.n_jobs,
                callbacks=all_callbacks,
                gc_after_trial=True,
            )

            # Get best results
            self.best_params = self.study.best_params
            self.best_score = self.study.best_value

            # Train final model with best parameters
            self.best_model = self._train_final_model(
                model_class, X_train, y_train, self.best_params, **kwargs
            )

            # Log results to MLflow
            if self.mlflow_tracking:
                mlflow.log_params(self.best_params)
                mlflow.log_metrics(
                    {
                        f"best_{self.metric}": self.best_score,
                        "n_trials_completed": len(self.study.trials),
                    }
                )
                mlflow.sklearn.log_model(self.best_model, "best_model")

            # Prepare results
            results = {
                "best_params": self.best_params,
                "best_score": self.best_score,
                "best_model": self.best_model,
                "study": self.study,
                "optimization_history": self._get_optimization_history(),
                "importance": self._get_parameter_importance(),
                "model_name": model_name,
            }

            logger.info(
                f"Optimization complete. Best {self.metric}: {self.best_score:.4f}"
            )
            logger.info(f"Best parameters: {self.best_params}")

            return results

        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            raise
        finally:
            if self.mlflow_tracking:
                mlflow.end_run()

    def _create_objective(
        self,
        model_class: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        search_space: Dict[str, Any],
        **kwargs,
    ) -> Callable:
        """Create objective function for Optuna."""

        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_config in search_space.items():
                param_type = param_config["type"]

                if param_type == "int":
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        step=param_config.get("step", 1),
                        log=param_config.get("log", False),
                    )
                elif param_type == "float":
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        step=param_config.get("step"),
                        log=param_config.get("log", False),
                    )
                elif param_type == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config["choices"]
                    )
                elif param_type == "conditional":
                    # Handle conditional parameters
                    condition_param = param_config["condition"]["param"]
                    condition_value = param_config["condition"]["value"]

                    if params.get(condition_param) == condition_value:
                        if param_config["conditional_type"] == "int":
                            params[param_name] = trial.suggest_int(
                                param_name, param_config["low"], param_config["high"]
                            )
                        elif param_config["conditional_type"] == "float":
                            params[param_name] = trial.suggest_float(
                                param_name, param_config["low"], param_config["high"]
                            )

            # Create model instance
            try:
                model = model_class(**{**params, **kwargs})
            except Exception as e:
                logger.warning(f"Failed to create model with params {params}: {str(e)}")
                return float("inf") if self.direction == "minimize" else float("-inf")

            # Evaluate model
            if X_val is not None and y_val is not None:
                # Use validation set
                model.fit(X_train, y_train)

                if hasattr(model, f"score_{self.metric}"):
                    score = getattr(model, f"score_{self.metric}")(X_val, y_val)
                else:
                    # Use sklearn scorer
                    scorer = make_scorer(
                        self._get_metric_function(),
                        greater_is_better=(self.direction == "maximize"),
                    )
                    score = scorer(model, X_val, y_val)
            else:
                # Use cross-validation
                scorer = make_scorer(
                    self._get_metric_function(),
                    greater_is_better=(self.direction == "maximize"),
                )

                scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=self.cv,
                    scoring=scorer,
                    n_jobs=1,  # Avoid nested parallelism
                )
                score = scores.mean()

            # Log to MLflow if enabled
            if self.mlflow_tracking:
                with mlflow.start_run(nested=True):
                    mlflow.log_params(params)
                    mlflow.log_metric(self.metric, score)

            # Track in optimization history
            self.optimization_history.append(
                {
                    "trial": trial.number,
                    "params": params,
                    "score": score,
                    "datetime": datetime.now(),
                }
            )

            return score

        return objective

    def _train_final_model(
        self,
        model_class: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        best_params: Dict[str, Any],
        **kwargs,
    ) -> Any:
        """Train final model with best parameters."""
        logger.info("Training final model with best parameters")

        model = model_class(**{**best_params, **kwargs})
        model.fit(X_train, y_train)

        return model

    def _get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as DataFrame."""
        if not self.study:
            return pd.DataFrame()

        history = []
        for trial in self.study.trials:
            record = {
                "trial": trial.number,
                "value": trial.value,
                "datetime": trial.datetime_complete,
                "duration": trial.duration.total_seconds() if trial.duration else None,
                "state": trial.state.name,
            }
            # Add parameters
            for param, value in trial.params.items():
                record[f"param_{param}"] = value

            history.append(record)

        return pd.DataFrame(history)

    def _get_parameter_importance(self) -> pd.DataFrame:
        """Get parameter importance using fANOVA."""
        if not self.study or len(self.study.trials) < 10:
            return pd.DataFrame()

        try:
            importance = optuna.importance.get_param_importances(
                self.study, evaluator=optuna.importance.FanovaImportanceEvaluator()
            )

            importance_df = pd.DataFrame(
                [
                    {"parameter": param, "importance": imp}
                    for param, imp in importance.items()
                ]
            ).sort_values("importance", ascending=False)

            return importance_df
        except Exception as e:
            logger.warning(f"Could not calculate parameter importance: {str(e)}")
            return pd.DataFrame()

    def _get_default_callbacks(self) -> List[Callable]:
        """Get default Optuna callbacks."""
        callbacks = []

        # Early stopping callback
        if settings.models.early_stopping_rounds:
            callbacks.append(
                optuna.pruners.MedianPruner(
                    n_startup_trials=5,
                    n_warmup_steps=settings.models.early_stopping_rounds,
                )
            )

        return callbacks

    def _get_model_class(self, model_name: str) -> Any:
        """Get model class from string name."""
        # Import model classes
        if self.task_type == "classification":
            from src.ml.models.classifier import ModelTrainer

            trainer = ModelTrainer()
            return trainer.models.get(model_name)
        else:
            from src.ml.models.regressor import RegressionTrainer

            trainer = RegressionTrainer()
            return trainer.models.get(model_name)

    def _get_metric_function(self) -> Callable:
        """Get metric function from metric name."""
        from sklearn import metrics

        metric_map = {
            # Classification
            "accuracy": metrics.accuracy_score,
            "precision": lambda y_true, y_pred: metrics.precision_score(
                y_true, y_pred, average="weighted"
            ),
            "recall": lambda y_true, y_pred: metrics.recall_score(
                y_true, y_pred, average="weighted"
            ),
            "f1": lambda y_true, y_pred: metrics.f1_score(
                y_true, y_pred, average="weighted"
            ),
            "roc_auc": metrics.roc_auc_score,
            "log_loss": metrics.log_loss,
            # Regression
            "mse": metrics.mean_squared_error,
            "rmse": lambda y_true, y_pred: np.sqrt(
                metrics.mean_squared_error(y_true, y_pred)
            ),
            "mae": metrics.mean_absolute_error,
            "r2": metrics.r2_score,
            "mape": metrics.mean_absolute_percentage_error,
            "explained_variance": metrics.explained_variance_score,
        }

        return metric_map.get(self.metric, metrics.accuracy_score)

    def plot_optimization_history(self) -> Any:
        """Plot optimization history using Optuna's visualization."""
        if not self.study:
            raise ValueError("No study available. Run optimization first.")

        try:
            import optuna.visualization as vis

            # Create multiple plots
            plots = {
                "history": vis.plot_optimization_history(self.study),
                "importance": vis.plot_param_importances(self.study),
                "parallel": vis.plot_parallel_coordinate(self.study),
                "slice": vis.plot_slice(self.study),
                "contour": vis.plot_contour(self.study),
            }

            return plots
        except ImportError:
            logger.warning(
                "Optuna visualization requires plotly. Install with: pip install plotly"
            )
            return None

    def save_study(self, filepath: str):
        """Save study to file."""
        if not self.study:
            raise ValueError("No study available to save.")

        joblib.dump(self.study, filepath)
        logger.info(f"Study saved to {filepath}")

    def load_study(self, filepath: str):
        """Load study from file."""
        self.study = joblib.load(filepath)
        logger.info(f"Study loaded from {filepath}")

        # Extract best results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

    def continue_optimization(
        self,
        model_class: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        additional_trials: int = 50,
    ) -> Dict[str, Any]:
        """Continue optimization with additional trials."""
        if not self.study:
            raise ValueError("No study available. Run optimization first.")

        logger.info(
            f"Continuing optimization with {additional_trials} additional trials"
        )

        # Update trial count
        original_trials = self.n_trials
        self.n_trials = additional_trials

        # Run additional optimization
        results = self.optimize(
            model_class=model_class, X_train=X_train, y_train=y_train
        )

        # Restore original trial count
        self.n_trials = original_trials

        return results


class MultiObjectiveOptimizer(OptunaOptimizer):
    """
    Multi-objective optimization using Optuna.

    Optimizes multiple metrics simultaneously using Pareto front.
    """

    def __init__(
        self,
        task_type: str = "classification",
        metrics: List[str] = None,
        directions: List[str] = None,
        **kwargs,
    ):
        """
        Initialize multi-objective optimizer.

        Args:
            task_type: Type of ML task
            metrics: List of metrics to optimize
            directions: List of optimization directions
            **kwargs: Additional arguments for OptunaOptimizer
        """
        super().__init__(task_type=task_type, **kwargs)

        self.metrics = metrics or ["accuracy", "f1"]
        self.directions = directions or ["maximize"] * len(self.metrics)

        if len(self.metrics) != len(self.directions):
            raise ValueError("Number of metrics must match number of directions")

    def optimize(
        self, model_class: Any, X_train: np.ndarray, y_train: np.ndarray, **kwargs
    ) -> Dict[str, Any]:
        """
        Perform multi-objective optimization.

        Returns:
            Dictionary with Pareto front solutions
        """
        logger.info(
            f"Starting multi-objective optimization for metrics: {self.metrics}"
        )

        # Create multi-objective study
        self.study = optuna.create_study(
            study_name=self.study_name,
            directions=self.directions,
            sampler=optuna.samplers.NSGAIISampler(seed=self.random_state),
        )

        # Create multi-objective function
        objective = self._create_multi_objective(
            model_class, X_train, y_train, **kwargs
        )

        # Run optimization
        self.study.optimize(
            objective, n_trials=self.n_trials, timeout=self.timeout, n_jobs=self.n_jobs
        )

        # Get Pareto front
        pareto_front = self._get_pareto_front()

        return {
            "pareto_front": pareto_front,
            "study": self.study,
            "best_trials": self.study.best_trials,
        }

    def _create_multi_objective(
        self, model_class: Any, X_train: np.ndarray, y_train: np.ndarray, **kwargs
    ) -> Callable:
        """Create multi-objective function."""

        def objective(trial):
            # Get parameters (reuse from parent class)
            search_space = self.search_space_builder.get_search_space(
                model_class.__name__, self.task_type
            )
            params = self._sample_parameters(trial, search_space)

            # Create and evaluate model
            model = model_class(**params)

            # Calculate all metrics
            scores = []
            for metric in self.metrics:
                scorer = make_scorer(
                    self._get_metric_function_by_name(metric),
                    greater_is_better=(
                        metric not in ["mse", "mae", "rmse", "log_loss"]
                    ),
                )

                cv_scores = cross_val_score(
                    model, X_train, y_train, cv=self.cv, scoring=scorer, n_jobs=1
                )
                scores.append(cv_scores.mean())

            return scores

        return objective

    def _get_pareto_front(self) -> pd.DataFrame:
        """Extract Pareto front solutions."""
        pareto_trials = []

        for trial in self.study.best_trials:
            record = {
                "trial": trial.number,
                **{f"metric_{m}": v for m, v in zip(self.metrics, trial.values)},
                **trial.params,
            }
            pareto_trials.append(record)

        return pd.DataFrame(pareto_trials)

    def _sample_parameters(self, trial, search_space):
        """Sample parameters from search space."""
        params = {}
        for param_name, param_config in search_space.items():
            if param_config["type"] == "int":
                params[param_name] = trial.suggest_int(
                    param_name, param_config["low"], param_config["high"]
                )
            elif param_config["type"] == "float":
                params[param_name] = trial.suggest_float(
                    param_name, param_config["low"], param_config["high"]
                )
            elif param_config["type"] == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name, param_config["choices"]
                )
        return params

    def _get_metric_function_by_name(self, metric_name: str) -> Callable:
        """Get metric function by name."""
        # Reuse from parent class
        self.metric = metric_name
        return self._get_metric_function()


# Convenience functions
def quick_optimize(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str = "classification",
    metric: str = None,
    n_trials: int = 50,
    **kwargs,
) -> Dict[str, Any]:
    """
    Quick optimization for a model.

    Args:
        model_name: Name of the model to optimize
        X_train: Training features
        y_train: Training labels
        task_type: "classification" or "regression"
        metric: Metric to optimize
        n_trials: Number of trials
        **kwargs: Additional arguments

    Returns:
        Optimization results
    """
    optimizer = OptunaOptimizer(
        task_type=task_type, metric=metric, n_trials=n_trials, **kwargs
    )

    return optimizer.optimize(model_name, X_train, y_train)


# Export public API
__all__ = ["OptunaOptimizer", "MultiObjectiveOptimizer", "quick_optimize"]
