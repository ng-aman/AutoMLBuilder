# src/ml/models/classifier.py
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """Handles classification model training and evaluation"""

    def __init__(self):
        self.models = {
            "RandomForestClassifier": RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            "XGBClassifier": XGBClassifier(
                n_estimators=100,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss",
            ),
            "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
            "GradientBoostingClassifier": GradientBoostingClassifier(
                n_estimators=100, random_state=42
            ),
            "SVC": SVC(random_state=42, probability=True),
            "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
            "KNeighborsClassifier": KNeighborsClassifier(),
            "GaussianNB": GaussianNB(),
        }

        self.trained_models = {}
        self.results = {}

    def train(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        model_name: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Train a single model

        Returns:
            Tuple of (trained_model, metrics)
        """
        logger.info(f"Training {model_name}")

        # Get model
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")

        model = self.models[model_name]

        # Update parameters if provided
        if params:
            model.set_params(**params)

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None

        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)

        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)

        # Store results
        self.trained_models[model_name] = model
        self.results[model_name] = metrics

        logger.info(
            f"{model_name} training complete. Accuracy: {metrics['accuracy']:.3f}"
        )

        return model, metrics

    def train_multiple(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        model_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Train multiple models and compare results"""
        if model_names is None:
            model_names = list(self.models.keys())

        results = {}

        for model_name in model_names:
            try:
                model, metrics = self.train(
                    X_train, y_train, X_test, y_test, model_name
                )
                results[model_name] = {"model": model, "metrics": metrics}
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")
                results[model_name] = {"model": None, "metrics": None, "error": str(e)}

        return results

    def cross_validate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        model_name: str,
        cv: int = 5,
        scoring: str = "accuracy",
    ) -> Dict[str, Any]:
        """Perform cross-validation"""
        logger.info(f"Cross-validating {model_name}")

        model = self.models[model_name]

        # Stratified K-Fold for classification
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        # Perform cross-validation
        scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)

        cv_results = {
            "scores": scores.tolist(),
            "mean": scores.mean(),
            "std": scores.std(),
            "scoring": scoring,
            "cv_folds": cv,
        }

        logger.info(
            f"{model_name} CV Score: {cv_results['mean']:.3f} "
            f"(+/- {cv_results['std'] * 2:.3f})"
        )

        return cv_results

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Calculate classification metrics"""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

        # Add ROC AUC if probabilities available
        if y_pred_proba is not None:
            try:
                # Handle binary and multiclass
                n_classes = y_pred_proba.shape[1]
                if n_classes == 2:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    # Multiclass ROC AUC
                    metrics["roc_auc"] = roc_auc_score(
                        y_true, y_pred_proba, multi_class="ovr", average="weighted"
                    )
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {str(e)}")
                metrics["roc_auc"] = None

        return metrics

    def get_feature_importance(
        self, model_name: str, feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Get feature importance for tree-based models"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")

        model = self.trained_models[model_name]

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_

            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(importances))]

            feature_importance = pd.DataFrame(
                {"feature": feature_names, "importance": importances}
            ).sort_values("importance", ascending=False)

            return feature_importance
        else:
            logger.warning(f"Model {model_name} does not support feature importance")
            return pd.DataFrame()

    def get_best_model(
        self, metric: str = "accuracy"
    ) -> Tuple[str, Any, Dict[str, float]]:
        """Get the best performing model"""
        if not self.results:
            raise ValueError("No models trained yet")

        best_model_name = max(
            self.results.keys(), key=lambda x: self.results[x].get(metric, 0)
        )

        return (
            best_model_name,
            self.trained_models[best_model_name],
            self.results[best_model_name],
        )

    def generate_classification_report(
        self,
        model_name: str,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        target_names: Optional[List[str]] = None,
    ) -> str:
        """Generate detailed classification report"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")

        model = self.trained_models[model_name]
        y_pred = model.predict(X_test)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Classification report
        report = classification_report(
            y_test, y_pred, target_names=target_names, output_dict=False
        )

        return f"Confusion Matrix:\n{cm}\n\nClassification Report:\n{report}"
