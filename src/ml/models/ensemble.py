# src/ml/models/ensemble.py
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union, Literal
from sklearn.ensemble import (
    VotingClassifier,
    VotingRegressor,
    StackingClassifier,
    StackingRegressor,
    BaggingClassifier,
    BaggingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.model_selection import cross_val_predict, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, r2_score
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
import warnings
from src.utils.logger import get_logger
from .classifier import ModelTrainer
from .regressor import RegressionTrainer

logger = get_logger(__name__)


class EnsembleBuilder:
    """Builds and manages ensemble models for classification and regression"""

    def __init__(
        self, task_type: Literal["classification", "regression"] = "classification"
    ):
        """
        Initialize ensemble builder

        Args:
            task_type: Type of ML task ("classification" or "regression")
        """
        self.task_type = task_type
        self.base_models = {}
        self.ensemble_models = {}
        self.results = {}

        # Initialize base model trainers
        if task_type == "classification":
            self.base_trainer = ModelTrainer()
            self.metric_name = "accuracy"
            self.cv_class = StratifiedKFold
        else:
            self.base_trainer = RegressionTrainer()
            self.metric_name = "r2"
            self.cv_class = KFold

    def train_base_models(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        model_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Train base models for ensemble"""
        logger.info(f"Training base models for {self.task_type}")

        # Train multiple base models
        results = self.base_trainer.train_multiple(
            X_train, y_train, X_test, y_test, model_names
        )

        # Store successfully trained models
        for name, result in results.items():
            if result["model"] is not None:
                self.base_models[name] = {
                    "model": result["model"],
                    "metrics": result["metrics"],
                }

        logger.info(f"Successfully trained {len(self.base_models)} base models")
        return self.base_models

    def create_voting_ensemble(
        self,
        estimators: Optional[List[Tuple[str, Any]]] = None,
        voting: str = "soft",
        weights: Optional[List[float]] = None,
    ) -> Union[VotingClassifier, VotingRegressor]:
        """
        Create voting ensemble

        Args:
            estimators: List of (name, model) tuples. If None, uses all base models
            voting: 'hard' or 'soft' (for classification only)
            weights: Weights for each model

        Returns:
            Voting ensemble model
        """
        if estimators is None:
            if not self.base_models:
                raise ValueError("No base models trained. Run train_base_models first.")
            estimators = [
                (name, data["model"]) for name, data in self.base_models.items()
            ]

        logger.info(f"Creating voting ensemble with {len(estimators)} models")

        if self.task_type == "classification":
            ensemble = VotingClassifier(
                estimators=estimators, voting=voting, weights=weights, n_jobs=-1
            )
        else:
            ensemble = VotingRegressor(
                estimators=estimators, weights=weights, n_jobs=-1
            )

        return ensemble

    def create_stacking_ensemble(
        self,
        estimators: Optional[List[Tuple[str, Any]]] = None,
        final_estimator: Optional[Any] = None,
        cv: int = 5,
        passthrough: bool = False,
    ) -> Union[StackingClassifier, StackingRegressor]:
        """
        Create stacking ensemble

        Args:
            estimators: List of (name, model) tuples. If None, uses all base models
            final_estimator: Meta-learner. If None, uses LogisticRegression or Ridge
            cv: Number of CV folds for generating meta-features
            passthrough: Whether to pass original features to meta-learner

        Returns:
            Stacking ensemble model
        """
        if estimators is None:
            if not self.base_models:
                raise ValueError("No base models trained. Run train_base_models first.")
            estimators = [
                (name, data["model"]) for name, data in self.base_models.items()
            ]

        if final_estimator is None:
            if self.task_type == "classification":
                final_estimator = LogisticRegression(random_state=42, max_iter=1000)
            else:
                final_estimator = Ridge(random_state=42)

        logger.info(f"Creating stacking ensemble with {len(estimators)} base models")

        if self.task_type == "classification":
            ensemble = StackingClassifier(
                estimators=estimators,
                final_estimator=final_estimator,
                cv=cv,
                passthrough=passthrough,
                n_jobs=-1,
            )
        else:
            ensemble = StackingRegressor(
                estimators=estimators,
                final_estimator=final_estimator,
                cv=cv,
                passthrough=passthrough,
                n_jobs=-1,
            )

        return ensemble

    def create_blending_ensemble(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray],
        estimators: Optional[List[Tuple[str, Any]]] = None,
        final_estimator: Optional[Any] = None,
    ) -> "BlendingEnsemble":
        """
        Create blending ensemble (similar to stacking but uses holdout validation)

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features for generating blend features
            y_val: Validation labels
            estimators: Base models
            final_estimator: Meta-learner

        Returns:
            Blending ensemble model
        """
        if estimators is None:
            if not self.base_models:
                raise ValueError("No base models trained. Run train_base_models first.")
            estimators = [
                (name, data["model"]) for name, data in self.base_models.items()
            ]

        if final_estimator is None:
            if self.task_type == "classification":
                final_estimator = LogisticRegression(random_state=42, max_iter=1000)
            else:
                final_estimator = LinearRegression()

        logger.info(f"Creating blending ensemble with {len(estimators)} models")

        blender = BlendingEnsemble(
            estimators=estimators,
            final_estimator=final_estimator,
            task_type=self.task_type,
        )

        # Train blending ensemble
        blender.fit(X_train, y_train, X_val, y_val)

        return blender

    def create_multi_level_ensemble(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        levels: List[Dict[str, Any]],
    ) -> "MultiLevelEnsemble":
        """
        Create multi-level ensemble (ensemble of ensembles)

        Args:
            X_train: Training features
            y_train: Training labels
            levels: List of level configurations
                   Each level: {'models': [...], 'ensemble_type': 'voting'/'stacking', ...}

        Returns:
            Multi-level ensemble model
        """
        logger.info(f"Creating multi-level ensemble with {len(levels)} levels")

        mle = MultiLevelEnsemble(task_type=self.task_type)
        mle.fit(X_train, y_train, levels)

        return mle

    def train_ensemble(
        self,
        ensemble_model: Any,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        ensemble_name: str,
    ) -> Tuple[Any, Dict[str, float]]:
        """Train and evaluate an ensemble model"""
        logger.info(f"Training ensemble: {ensemble_name}")

        # Train ensemble
        ensemble_model.fit(X_train, y_train)

        # Make predictions
        y_pred = ensemble_model.predict(X_test)

        # Calculate metrics
        if self.task_type == "classification":
            metrics = self.base_trainer._calculate_metrics(
                y_test,
                y_pred,
                (
                    ensemble_model.predict_proba(X_test)
                    if hasattr(ensemble_model, "predict_proba")
                    else None
                ),
            )
        else:
            metrics = self.base_trainer._calculate_metrics(y_test, y_pred)

        # Store results
        self.ensemble_models[ensemble_name] = ensemble_model
        self.results[ensemble_name] = metrics

        logger.info(
            f"{ensemble_name} training complete. {self.metric_name}: {metrics[self.metric_name]:.3f}"
        )

        return ensemble_model, metrics

    def compare_all_models(self) -> pd.DataFrame:
        """Compare performance of all models (base + ensembles)"""
        all_results = {}

        # Add base model results
        for name, data in self.base_models.items():
            all_results[f"Base_{name}"] = data["metrics"]

        # Add ensemble results
        for name, metrics in self.results.items():
            all_results[f"Ensemble_{name}"] = metrics

        # Convert to DataFrame
        comparison_df = pd.DataFrame(all_results).T

        # Sort by primary metric
        comparison_df = comparison_df.sort_values(self.metric_name, ascending=False)

        return comparison_df

    def get_ensemble_weights(self, ensemble_name: str) -> Dict[str, Any]:
        """Get weights/importance of base models in an ensemble"""
        if ensemble_name not in self.ensemble_models:
            raise ValueError(f"Ensemble {ensemble_name} not found")

        ensemble = self.ensemble_models[ensemble_name]
        weights_info = {}

        if isinstance(ensemble, (VotingClassifier, VotingRegressor)):
            if hasattr(ensemble, "weights") and ensemble.weights is not None:
                for i, (name, _) in enumerate(ensemble.estimators):
                    weights_info[name] = ensemble.weights[i]
            else:
                # Equal weights
                n_estimators = len(ensemble.estimators)
                for name, _ in ensemble.estimators:
                    weights_info[name] = 1.0 / n_estimators

        elif isinstance(ensemble, (StackingClassifier, StackingRegressor)):
            # Get coefficients from meta-learner if linear
            if hasattr(ensemble.final_estimator_, "coef_"):
                coef = ensemble.final_estimator_.coef_
                if coef.ndim == 1:
                    for i, (name, _) in enumerate(ensemble.estimators):
                        weights_info[name] = coef[i]
                else:
                    # Multi-class case
                    for i, (name, _) in enumerate(ensemble.estimators):
                        weights_info[name] = np.mean(np.abs(coef[:, i]))

        return weights_info


class BlendingEnsemble(BaseEstimator):
    """Custom blending ensemble implementation"""

    def __init__(
        self,
        estimators: List[Tuple[str, Any]],
        final_estimator: Any,
        task_type: str = "classification",
    ):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.task_type = task_type
        self.base_models_ = []
        self.meta_model_ = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ):
        """Fit blending ensemble"""
        # Train base models on training data
        self.base_models_ = []
        blend_train = []

        for name, estimator in self.estimators:
            logger.info(f"Training base model: {name}")
            model = clone(estimator)
            model.fit(X_train, y_train)
            self.base_models_.append((name, model))

            # Generate predictions on validation set
            if self.task_type == "classification" and hasattr(model, "predict_proba"):
                pred = model.predict_proba(X_val)[:, 1] if pred.shape[1] == 2 else pred
            else:
                pred = model.predict(X_val)

            blend_train.append(pred)

        # Create blend features
        blend_train = np.column_stack(blend_train)

        # Train meta-model
        logger.info("Training meta-model on blend features")
        self.meta_model_ = clone(self.final_estimator)
        self.meta_model_.fit(blend_train, y_val)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        # Generate base model predictions
        blend_test = []

        for name, model in self.base_models_:
            if self.task_type == "classification" and hasattr(model, "predict_proba"):
                pred = model.predict_proba(X)
                pred = pred[:, 1] if pred.shape[1] == 2 else pred
            else:
                pred = model.predict(X)

            blend_test.append(pred)

        # Create blend features
        blend_test = np.column_stack(blend_test)

        # Meta-model prediction
        return self.meta_model_.predict(blend_test)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (classification only)"""
        if self.task_type != "classification":
            raise AttributeError("predict_proba is only available for classification")

        # Generate base model predictions
        blend_test = []

        for name, model in self.base_models_:
            if hasattr(model, "predict_proba"):
                pred = model.predict_proba(X)
                pred = pred[:, 1] if pred.shape[1] == 2 else pred
            else:
                pred = model.predict(X)

            blend_test.append(pred)

        blend_test = np.column_stack(blend_test)

        # Meta-model prediction
        if hasattr(self.meta_model_, "predict_proba"):
            return self.meta_model_.predict_proba(blend_test)
        else:
            # Convert predictions to probabilities
            pred = self.meta_model_.predict(blend_test)
            return np.column_stack([1 - pred, pred])


class MultiLevelEnsemble(BaseEstimator):
    """Multi-level ensemble (ensemble of ensembles)"""

    def __init__(self, task_type: str = "classification"):
        self.task_type = task_type
        self.levels_ = []

    def fit(self, X: np.ndarray, y: np.ndarray, levels: List[Dict[str, Any]]):
        """
        Fit multi-level ensemble

        Args:
            X: Training features
            y: Training labels
            levels: List of level configurations
        """
        current_X = X

        for i, level_config in enumerate(levels):
            logger.info(f"Training level {i + 1}")

            ensemble_type = level_config.get("ensemble_type", "voting")
            models = level_config.get("models", [])

            # Create ensemble for this level
            if ensemble_type == "voting":
                if self.task_type == "classification":
                    ensemble = VotingClassifier(
                        estimators=models,
                        voting=level_config.get("voting", "soft"),
                        weights=level_config.get("weights"),
                        n_jobs=-1,
                    )
                else:
                    ensemble = VotingRegressor(
                        estimators=models,
                        weights=level_config.get("weights"),
                        n_jobs=-1,
                    )

            elif ensemble_type == "stacking":
                final_est = level_config.get("final_estimator")
                if final_est is None:
                    final_est = (
                        LogisticRegression()
                        if self.task_type == "classification"
                        else Ridge()
                    )

                if self.task_type == "classification":
                    ensemble = StackingClassifier(
                        estimators=models,
                        final_estimator=final_est,
                        cv=level_config.get("cv", 5),
                        n_jobs=-1,
                    )
                else:
                    ensemble = StackingRegressor(
                        estimators=models,
                        final_estimator=final_est,
                        cv=level_config.get("cv", 5),
                        n_jobs=-1,
                    )

            else:
                raise ValueError(f"Unknown ensemble type: {ensemble_type}")

            # Fit ensemble
            ensemble.fit(current_X, y)
            self.levels_.append(ensemble)

            # If not the last level, generate predictions for next level
            if i < len(levels) - 1:
                # Use cross-validation predictions as features for next level
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                if hasattr(ensemble, "predict_proba"):
                    current_X = cross_val_predict(
                        ensemble, current_X, y, cv=cv, method="predict_proba"
                    )
                else:
                    current_X = cross_val_predict(
                        ensemble, current_X, y, cv=cv
                    ).reshape(-1, 1)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions through all levels"""
        current_X = X

        for ensemble in self.levels_:
            if hasattr(ensemble, "predict_proba") and ensemble != self.levels_[-1]:
                # Use probabilities as features for next level (except last)
                current_X = ensemble.predict_proba(current_X)
            else:
                current_X = ensemble.predict(current_X)
                if current_X.ndim == 1 and ensemble != self.levels_[-1]:
                    current_X = current_X.reshape(-1, 1)

        return current_X

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (classification only)"""
        if self.task_type != "classification":
            raise AttributeError("predict_proba is only available for classification")

        current_X = X

        for i, ensemble in enumerate(self.levels_):
            if i < len(self.levels_) - 1:
                # Intermediate levels
                if hasattr(ensemble, "predict_proba"):
                    current_X = ensemble.predict_proba(current_X)
                else:
                    current_X = ensemble.predict(current_X).reshape(-1, 1)
            else:
                # Last level
                if hasattr(ensemble, "predict_proba"):
                    return ensemble.predict_proba(current_X)
                else:
                    pred = ensemble.predict(current_X)
                    return np.column_stack([1 - pred, pred])


# Convenience function for quick ensemble creation
def create_auto_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str = "classification",
    ensemble_types: List[str] = ["voting", "stacking"],
) -> Dict[str, Any]:
    """
    Automatically create and compare multiple ensemble types

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        task_type: "classification" or "regression"
        ensemble_types: List of ensemble types to create

    Returns:
        Dictionary with results and best ensemble
    """
    builder = EnsembleBuilder(task_type=task_type)

    # Train base models
    builder.train_base_models(X_train, y_train, X_test, y_test)

    results = {}

    # Create different ensemble types
    if "voting" in ensemble_types:
        voting_ens = builder.create_voting_ensemble()
        model, metrics = builder.train_ensemble(
            voting_ens, X_train, y_train, X_test, y_test, "voting"
        )
        results["voting"] = {"model": model, "metrics": metrics}

    if "stacking" in ensemble_types:
        stacking_ens = builder.create_stacking_ensemble()
        model, metrics = builder.train_ensemble(
            stacking_ens, X_train, y_train, X_test, y_test, "stacking"
        )
        results["stacking"] = {"model": model, "metrics": metrics}

    # Compare all models
    comparison = builder.compare_all_models()

    # Get best model
    best_name = comparison.index[0]
    if best_name.startswith("Ensemble_"):
        best_model = builder.ensemble_models[best_name.replace("Ensemble_", "")]
    else:
        best_model = builder.base_models[best_name.replace("Base_", "")]["model"]

    return {
        "results": results,
        "comparison": comparison,
        "best_model": best_model,
        "best_name": best_name,
        "builder": builder,
    }
