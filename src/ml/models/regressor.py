# src/ml/models/regressor.py
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    HuberRegressor,
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score,
    max_error,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RegressionTrainer:
    """Handles regression model training and evaluation"""

    def __init__(self):
        self.models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(random_state=42),
            "Lasso": Lasso(random_state=42, max_iter=2000),
            "ElasticNet": ElasticNet(random_state=42, max_iter=2000),
            "RandomForestRegressor": RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            "XGBRegressor": XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "LGBMRegressor": LGBMRegressor(
                n_estimators=100, random_state=42, n_jobs=-1, verbose=-1
            ),
            "CatBoostRegressor": CatBoostRegressor(
                iterations=100, random_state=42, verbose=False
            ),
            "GradientBoostingRegressor": GradientBoostingRegressor(
                n_estimators=100, random_state=42
            ),
            "SVR": SVR(kernel="rbf"),
            "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
            "KNeighborsRegressor": KNeighborsRegressor(),
            "AdaBoostRegressor": AdaBoostRegressor(n_estimators=100, random_state=42),
            "HuberRegressor": HuberRegressor(max_iter=1000),
        }

        self.trained_models = {}
        self.results = {}
        self.predictions = {}

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
        Train a single regression model

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
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            raise

        # Make predictions
        y_pred = model.predict(X_test)

        # Store predictions for later analysis
        self.predictions[model_name] = {"y_true": y_test, "y_pred": y_pred}

        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred)

        # Store results
        self.trained_models[model_name] = model
        self.results[model_name] = metrics

        logger.info(
            f"{model_name} training complete. R² Score: {metrics['r2']:.3f}, "
            f"RMSE: {metrics['rmse']:.3f}"
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
            # Use a default subset for faster training
            model_names = [
                "LinearRegression",
                "Ridge",
                "RandomForestRegressor",
                "XGBRegressor",
                "GradientBoostingRegressor",
            ]

        results = {}

        for model_name in model_names:
            try:
                model, metrics = self.train(
                    X_train, y_train, X_test, y_test, model_name
                )
                results[model_name] = {
                    "model": model,
                    "metrics": metrics,
                    "predictions": self.predictions[model_name],
                }
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
        scoring: str = "r2",
    ) -> Dict[str, Any]:
        """Perform cross-validation"""
        logger.info(f"Cross-validating {model_name}")

        model = self.models[model_name]

        # K-Fold for regression
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)

        # Perform cross-validation
        scores = cross_val_score(model, X, y, cv=kf, scoring=scoring, n_jobs=-1)

        cv_results = {
            "scores": scores.tolist(),
            "mean": scores.mean(),
            "std": scores.std(),
            "scoring": scoring,
            "cv_folds": cv,
        }

        logger.info(
            f"{model_name} CV Score ({scoring}): {cv_results['mean']:.3f} "
            f"(+/- {cv_results['std'] * 2:.3f})"
        )

        return cv_results

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate regression metrics"""

        # Basic metrics
        metrics = {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "explained_variance": explained_variance_score(y_true, y_pred),
            "max_error": max_error(y_true, y_pred),
        }

        # MAPE (handle zero values)
        try:
            if np.any(y_true == 0):
                # Use masked array for MAPE calculation
                mask = y_true != 0
                if np.any(mask):
                    metrics["mape"] = (
                        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
                        * 100
                    )
                else:
                    metrics["mape"] = np.inf
            else:
                metrics["mape"] = mean_absolute_percentage_error(y_true, y_pred) * 100
        except Exception as e:
            logger.warning(f"Could not calculate MAPE: {str(e)}")
            metrics["mape"] = None

        # Additional statistics
        residuals = y_true - y_pred
        metrics["mean_residual"] = np.mean(residuals)
        metrics["std_residual"] = np.std(residuals)

        # Adjusted R²
        n = len(y_true)
        if hasattr(self, "_n_features"):
            p = self._n_features
            if n > p + 1:
                metrics["adjusted_r2"] = 1 - (1 - metrics["r2"]) * (n - 1) / (n - p - 1)
            else:
                metrics["adjusted_r2"] = None

        return metrics

    def get_feature_importance(
        self, model_name: str, feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Get feature importance for tree-based models"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")

        model = self.trained_models[model_name]

        # Tree-based models
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_

            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(importances))]

            feature_importance = pd.DataFrame(
                {"feature": feature_names, "importance": importances}
            ).sort_values("importance", ascending=False)

            return feature_importance

        # Linear models with coefficients
        elif hasattr(model, "coef_"):
            coefficients = model.coef_

            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(coefficients))]

            feature_importance = pd.DataFrame(
                {"feature": feature_names, "coefficient": coefficients}
            ).sort_values("coefficient", ascending=False, key=abs)

            return feature_importance

        else:
            logger.warning(f"Model {model_name} does not support feature importance")
            return pd.DataFrame()

    def get_best_model(
        self, metric: str = "r2", minimize: bool = False
    ) -> Tuple[str, Any, Dict[str, float]]:
        """Get the best performing model"""
        if not self.results:
            raise ValueError("No models trained yet")

        if minimize:
            best_model_name = min(
                self.results.keys(),
                key=lambda x: self.results[x].get(metric, float("inf")),
            )
        else:
            best_model_name = max(
                self.results.keys(),
                key=lambda x: self.results[x].get(metric, float("-inf")),
            )

        return (
            best_model_name,
            self.trained_models[best_model_name],
            self.results[best_model_name],
        )

    def get_residual_analysis(self, model_name: str) -> Dict[str, Any]:
        """Perform residual analysis for a trained model"""
        if model_name not in self.predictions:
            raise ValueError(f"No predictions found for {model_name}")

        y_true = self.predictions[model_name]["y_true"]
        y_pred = self.predictions[model_name]["y_pred"]
        residuals = y_true - y_pred

        # Residual statistics
        analysis = {
            "residuals": residuals,
            "mean_residual": np.mean(residuals),
            "std_residual": np.std(residuals),
            "min_residual": np.min(residuals),
            "max_residual": np.max(residuals),
            "residual_skewness": self._calculate_skewness(residuals),
            "residual_kurtosis": self._calculate_kurtosis(residuals),
            "durbin_watson": self._durbin_watson(residuals),
        }

        # Check for heteroscedasticity
        analysis["heteroscedasticity"] = self._check_heteroscedasticity(
            y_pred, residuals
        )

        return analysis

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        from scipy.stats import skew

        return skew(data)

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        from scipy.stats import kurtosis

        return kurtosis(data)

    def _durbin_watson(self, residuals: np.ndarray) -> float:
        """Calculate Durbin-Watson statistic"""
        diff_resid = np.diff(residuals)
        return np.sum(diff_resid**2) / np.sum(residuals**2)

    def _check_heteroscedasticity(
        self, y_pred: np.ndarray, residuals: np.ndarray
    ) -> Dict[str, Any]:
        """Check for heteroscedasticity using Breusch-Pagan test"""
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            from statsmodels.regression.linear_model import OLS
            from statsmodels.tools.tools import add_constant

            # Prepare data for test
            X = add_constant(y_pred.reshape(-1, 1))

            # Perform Breusch-Pagan test
            lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(residuals, X)

            return {
                "test_statistic": lm,
                "p_value": lm_pvalue,
                "is_heteroscedastic": lm_pvalue < 0.05,
                "interpretation": (
                    "Heteroscedasticity detected"
                    if lm_pvalue < 0.05
                    else "No significant heteroscedasticity"
                ),
            }
        except ImportError:
            logger.warning(
                "statsmodels not installed. Skipping heteroscedasticity test."
            )
            return None

    def generate_regression_report(
        self,
        model_name: str,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        feature_names: Optional[List[str]] = None,
    ) -> str:
        """Generate detailed regression report"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")

        model = self.trained_models[model_name]
        y_pred = model.predict(X_test)

        # Get metrics
        metrics = self._calculate_metrics(y_test, y_pred)

        # Build report
        report = f"=== Regression Report for {model_name} ===\n\n"

        # Model metrics
        report += "Performance Metrics:\n"
        report += f"  R² Score:           {metrics['r2']:.4f}\n"
        report += f"  RMSE:               {metrics['rmse']:.4f}\n"
        report += f"  MAE:                {metrics['mae']:.4f}\n"
        if metrics.get("mape") is not None:
            report += f"  MAPE:               {metrics['mape']:.2f}%\n"
        report += f"  Explained Variance: {metrics['explained_variance']:.4f}\n"
        report += f"  Max Error:          {metrics['max_error']:.4f}\n"

        # Residual analysis
        residual_analysis = self.get_residual_analysis(model_name)
        report += "\nResidual Analysis:\n"
        report += f"  Mean Residual:      {residual_analysis['mean_residual']:.4f}\n"
        report += f"  Std Residual:       {residual_analysis['std_residual']:.4f}\n"
        report += f"  Durbin-Watson:      {residual_analysis['durbin_watson']:.4f}\n"

        # Feature importance (if available)
        feature_importance = self.get_feature_importance(model_name, feature_names)
        if not feature_importance.empty:
            report += "\nTop 10 Feature Importance:\n"
            for idx, row in feature_importance.head(10).iterrows():
                if "importance" in feature_importance.columns:
                    report += f"  {row['feature']:20s} {row['importance']:.4f}\n"
                else:
                    report += f"  {row['feature']:20s} {row['coefficient']:.4f}\n"

        return report
