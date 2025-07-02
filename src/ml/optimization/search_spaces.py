# src/ml/optimization/search_spaces.py
from typing import Dict, Any, List, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SearchSpaceBuilder:
    """
    Defines hyperparameter search spaces for different ML models.

    Provides optimized search spaces based on best practices and
    supports both classification and regression tasks.
    """

    def __init__(self):
        """Initialize search space builder."""
        self.search_spaces = self._initialize_search_spaces()

    def _initialize_search_spaces(self) -> Dict[str, Dict[str, Any]]:
        """Initialize all search spaces."""
        return {
            # Classification models
            "RandomForestClassifier": self._random_forest_classifier_space(),
            "XGBClassifier": self._xgboost_classifier_space(),
            "LogisticRegression": self._logistic_regression_space(),
            "GradientBoostingClassifier": self._gradient_boosting_classifier_space(),
            "SVC": self._svc_space(),
            "DecisionTreeClassifier": self._decision_tree_classifier_space(),
            "KNeighborsClassifier": self._knn_classifier_space(),
            "GaussianNB": self._gaussian_nb_space(),
            # Regression models
            "RandomForestRegressor": self._random_forest_regressor_space(),
            "XGBRegressor": self._xgboost_regressor_space(),
            "LinearRegression": self._linear_regression_space(),
            "Ridge": self._ridge_space(),
            "Lasso": self._lasso_space(),
            "ElasticNet": self._elastic_net_space(),
            "GradientBoostingRegressor": self._gradient_boosting_regressor_space(),
            "SVR": self._svr_space(),
            "DecisionTreeRegressor": self._decision_tree_regressor_space(),
            "KNeighborsRegressor": self._knn_regressor_space(),
            "LGBMRegressor": self._lgbm_regressor_space(),
            "CatBoostRegressor": self._catboost_regressor_space(),
            "AdaBoostRegressor": self._adaboost_regressor_space(),
            "HuberRegressor": self._huber_regressor_space(),
            # Ensemble models
            "VotingClassifier": self._voting_classifier_space(),
            "VotingRegressor": self._voting_regressor_space(),
            "StackingClassifier": self._stacking_classifier_space(),
            "StackingRegressor": self._stacking_regressor_space(),
        }

    def get_search_space(
        self,
        model_name: str,
        task_type: Optional[str] = None,
        custom_ranges: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Get search space for a specific model.

        Args:
            model_name: Name of the model
            task_type: Optional task type for validation
            custom_ranges: Optional custom parameter ranges to override defaults

        Returns:
            Dictionary defining the search space
        """
        if model_name not in self.search_spaces:
            logger.warning(
                f"No predefined search space for {model_name}, using empty space"
            )
            return {}

        search_space = self.search_spaces[model_name].copy()

        # Apply custom ranges if provided
        if custom_ranges:
            for param, config in custom_ranges.items():
                if param in search_space:
                    search_space[param].update(config)
                else:
                    search_space[param] = config

        return search_space

    # Random Forest search spaces
    def _random_forest_classifier_space(self) -> Dict[str, Any]:
        """Random Forest Classifier search space."""
        return {
            "n_estimators": {
                "type": "int",
                "low": 50,
                "high": 500,
                "step": 50,
                "log": False,
            },
            "max_depth": {"type": "int", "low": 3, "high": 20, "step": 1},
            "min_samples_split": {"type": "int", "low": 2, "high": 20, "step": 1},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10, "step": 1},
            "max_features": {
                "type": "categorical",
                "choices": ["sqrt", "log2", 0.5, 0.7, 0.9],
            },
            "bootstrap": {"type": "categorical", "choices": [True, False]},
            "criterion": {"type": "categorical", "choices": ["gini", "entropy"]},
        }

    def _random_forest_regressor_space(self) -> Dict[str, Any]:
        """Random Forest Regressor search space."""
        space = self._random_forest_classifier_space().copy()
        space["criterion"] = {
            "type": "categorical",
            "choices": ["squared_error", "absolute_error", "poisson"],
        }
        return space

    # XGBoost search spaces
    def _xgboost_classifier_space(self) -> Dict[str, Any]:
        """XGBoost Classifier search space."""
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 50},
            "max_depth": {"type": "int", "low": 3, "high": 15, "step": 1},
            "learning_rate": {
                "type": "float",
                "low": 0.01,
                "high": 0.3,
                "step": 0.01,
                "log": True,
            },
            "subsample": {"type": "float", "low": 0.6, "high": 1.0, "step": 0.1},
            "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0, "step": 0.1},
            "gamma": {"type": "float", "low": 0.0, "high": 1.0, "step": 0.1},
            "reg_alpha": {"type": "float", "low": 0.0, "high": 1.0, "step": 0.1},
            "reg_lambda": {"type": "float", "low": 0.0, "high": 1.0, "step": 0.1},
            "min_child_weight": {"type": "int", "low": 1, "high": 10, "step": 1},
        }

    def _xgboost_regressor_space(self) -> Dict[str, Any]:
        """XGBoost Regressor search space."""
        return self._xgboost_classifier_space()

    # LightGBM search spaces
    def _lgbm_regressor_space(self) -> Dict[str, Any]:
        """LightGBM Regressor search space."""
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 50},
            "max_depth": {"type": "int", "low": 3, "high": 15, "step": 1},
            "learning_rate": {
                "type": "float",
                "low": 0.01,
                "high": 0.3,
                "step": 0.01,
                "log": True,
            },
            "num_leaves": {"type": "int", "low": 20, "high": 300, "step": 10},
            "feature_fraction": {"type": "float", "low": 0.5, "high": 1.0, "step": 0.1},
            "bagging_fraction": {"type": "float", "low": 0.5, "high": 1.0, "step": 0.1},
            "bagging_freq": {"type": "int", "low": 0, "high": 10, "step": 1},
            "reg_alpha": {"type": "float", "low": 0.0, "high": 1.0, "step": 0.1},
            "reg_lambda": {"type": "float", "low": 0.0, "high": 1.0, "step": 0.1},
            "min_child_samples": {"type": "int", "low": 5, "high": 50, "step": 5},
        }

    # CatBoost search spaces
    def _catboost_regressor_space(self) -> Dict[str, Any]:
        """CatBoost Regressor search space."""
        return {
            "iterations": {"type": "int", "low": 50, "high": 500, "step": 50},
            "depth": {"type": "int", "low": 4, "high": 10, "step": 1},
            "learning_rate": {
                "type": "float",
                "low": 0.01,
                "high": 0.3,
                "step": 0.01,
                "log": True,
            },
            "l2_leaf_reg": {"type": "float", "low": 1.0, "high": 10.0, "step": 0.5},
            "border_count": {"type": "int", "low": 32, "high": 255, "step": 32},
            "bagging_temperature": {
                "type": "float",
                "low": 0.0,
                "high": 1.0,
                "step": 0.1,
            },
        }

    # Linear model search spaces
    def _logistic_regression_space(self) -> Dict[str, Any]:
        """Logistic Regression search space."""
        return {
            "C": {"type": "float", "low": 0.001, "high": 100.0, "log": True},
            "penalty": {
                "type": "categorical",
                "choices": ["l1", "l2", "elasticnet", "none"],
            },
            "solver": {
                "type": "conditional",
                "condition": {"param": "penalty", "value": "l1"},
                "conditional_type": "categorical",
                "choices": ["liblinear", "saga"],
            },
            "l1_ratio": {
                "type": "conditional",
                "condition": {"param": "penalty", "value": "elasticnet"},
                "conditional_type": "float",
                "low": 0.0,
                "high": 1.0,
                "step": 0.1,
            },
            "class_weight": {"type": "categorical", "choices": [None, "balanced"]},
        }

    def _linear_regression_space(self) -> Dict[str, Any]:
        """Linear Regression search space (minimal)."""
        return {
            "fit_intercept": {"type": "categorical", "choices": [True, False]},
            "normalize": {"type": "categorical", "choices": [True, False]},
        }

    def _ridge_space(self) -> Dict[str, Any]:
        """Ridge Regression search space."""
        return {
            "alpha": {"type": "float", "low": 0.001, "high": 100.0, "log": True},
            "fit_intercept": {"type": "categorical", "choices": [True, False]},
            "solver": {
                "type": "categorical",
                "choices": [
                    "auto",
                    "svd",
                    "cholesky",
                    "lsqr",
                    "sparse_cg",
                    "sag",
                    "saga",
                ],
            },
        }

    def _lasso_space(self) -> Dict[str, Any]:
        """Lasso Regression search space."""
        return {
            "alpha": {"type": "float", "low": 0.0001, "high": 10.0, "log": True},
            "fit_intercept": {"type": "categorical", "choices": [True, False]},
            "selection": {"type": "categorical", "choices": ["cyclic", "random"]},
        }

    def _elastic_net_space(self) -> Dict[str, Any]:
        """ElasticNet search space."""
        return {
            "alpha": {"type": "float", "low": 0.0001, "high": 10.0, "log": True},
            "l1_ratio": {"type": "float", "low": 0.0, "high": 1.0, "step": 0.1},
            "fit_intercept": {"type": "categorical", "choices": [True, False]},
            "selection": {"type": "categorical", "choices": ["cyclic", "random"]},
        }

    # Gradient Boosting search spaces
    def _gradient_boosting_classifier_space(self) -> Dict[str, Any]:
        """Gradient Boosting Classifier search space."""
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 300, "step": 50},
            "max_depth": {"type": "int", "low": 3, "high": 10, "step": 1},
            "learning_rate": {
                "type": "float",
                "low": 0.01,
                "high": 0.3,
                "step": 0.01,
                "log": True,
            },
            "subsample": {"type": "float", "low": 0.7, "high": 1.0, "step": 0.1},
            "min_samples_split": {"type": "int", "low": 2, "high": 20, "step": 2},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10, "step": 1},
            "max_features": {
                "type": "categorical",
                "choices": ["sqrt", "log2", 0.5, 0.7, 0.9],
            },
        }

    def _gradient_boosting_regressor_space(self) -> Dict[str, Any]:
        """Gradient Boosting Regressor search space."""
        space = self._gradient_boosting_classifier_space().copy()
        space["loss"] = {
            "type": "categorical",
            "choices": ["squared_error", "absolute_error", "huber", "quantile"],
        }
        return space

    # SVM search spaces
    def _svc_space(self) -> Dict[str, Any]:
        """SVC search space."""
        return {
            "C": {"type": "float", "low": 0.1, "high": 100.0, "log": True},
            "kernel": {
                "type": "categorical",
                "choices": ["linear", "poly", "rbf", "sigmoid"],
            },
            "gamma": {
                "type": "categorical",
                "choices": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
            },
            "degree": {
                "type": "conditional",
                "condition": {"param": "kernel", "value": "poly"},
                "conditional_type": "int",
                "low": 2,
                "high": 5,
            },
            "class_weight": {"type": "categorical", "choices": [None, "balanced"]},
        }

    def _svr_space(self) -> Dict[str, Any]:
        """SVR search space."""
        return {
            "C": {"type": "float", "low": 0.1, "high": 100.0, "log": True},
            "kernel": {
                "type": "categorical",
                "choices": ["linear", "poly", "rbf", "sigmoid"],
            },
            "gamma": {
                "type": "categorical",
                "choices": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
            },
            "epsilon": {"type": "float", "low": 0.01, "high": 1.0, "log": True},
            "degree": {
                "type": "conditional",
                "condition": {"param": "kernel", "value": "poly"},
                "conditional_type": "int",
                "low": 2,
                "high": 5,
            },
        }

    # Tree-based search spaces
    def _decision_tree_classifier_space(self) -> Dict[str, Any]:
        """Decision Tree Classifier search space."""
        return {
            "max_depth": {"type": "int", "low": 2, "high": 20, "step": 1},
            "min_samples_split": {"type": "int", "low": 2, "high": 20, "step": 1},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10, "step": 1},
            "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]},
            "criterion": {"type": "categorical", "choices": ["gini", "entropy"]},
            "splitter": {"type": "categorical", "choices": ["best", "random"]},
        }

    def _decision_tree_regressor_space(self) -> Dict[str, Any]:
        """Decision Tree Regressor search space."""
        space = self._decision_tree_classifier_space().copy()
        space["criterion"] = {
            "type": "categorical",
            "choices": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
        }
        return space

    # KNN search spaces
    def _knn_classifier_space(self) -> Dict[str, Any]:
        """KNN Classifier search space."""
        return {
            "n_neighbors": {"type": "int", "low": 3, "high": 50, "step": 2},
            "weights": {"type": "categorical", "choices": ["uniform", "distance"]},
            "algorithm": {
                "type": "categorical",
                "choices": ["auto", "ball_tree", "kd_tree", "brute"],
            },
            "leaf_size": {"type": "int", "low": 10, "high": 50, "step": 10},
            "p": {
                "type": "categorical",
                "choices": [1, 2],  # Manhattan or Euclidean distance
            },
        }

    def _knn_regressor_space(self) -> Dict[str, Any]:
        """KNN Regressor search space."""
        return self._knn_classifier_space()

    # Naive Bayes search space
    def _gaussian_nb_space(self) -> Dict[str, Any]:
        """Gaussian Naive Bayes search space."""
        return {
            "var_smoothing": {"type": "float", "low": 1e-10, "high": 1e-6, "log": True}
        }

    # AdaBoost search spaces
    def _adaboost_regressor_space(self) -> Dict[str, Any]:
        """AdaBoost Regressor search space."""
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 300, "step": 50},
            "learning_rate": {
                "type": "float",
                "low": 0.01,
                "high": 2.0,
                "step": 0.1,
                "log": True,
            },
            "loss": {
                "type": "categorical",
                "choices": ["linear", "square", "exponential"],
            },
        }

    # Huber Regressor search space
    def _huber_regressor_space(self) -> Dict[str, Any]:
        """Huber Regressor search space."""
        return {
            "epsilon": {"type": "float", "low": 1.0, "high": 2.0, "step": 0.1},
            "alpha": {"type": "float", "low": 0.0001, "high": 1.0, "log": True},
            "fit_intercept": {"type": "categorical", "choices": [True, False]},
        }

    # Ensemble search spaces
    def _voting_classifier_space(self) -> Dict[str, Any]:
        """Voting Classifier search space."""
        return {"voting": {"type": "categorical", "choices": ["soft", "hard"]}}

    def _voting_regressor_space(self) -> Dict[str, Any]:
        """Voting Regressor search space (minimal)."""
        return {}

    def _stacking_classifier_space(self) -> Dict[str, Any]:
        """Stacking Classifier search space."""
        return {
            "cv": {"type": "int", "low": 3, "high": 10, "step": 1},
            "passthrough": {"type": "categorical", "choices": [True, False]},
        }

    def _stacking_regressor_space(self) -> Dict[str, Any]:
        """Stacking Regressor search space."""
        return self._stacking_classifier_space()

    def get_compact_space(self, model_name: str) -> Dict[str, Any]:
        """
        Get a compact search space with fewer parameters for faster optimization.

        Args:
            model_name: Name of the model

        Returns:
            Compact search space
        """
        full_space = self.get_search_space(model_name)

        # Define key parameters for each model type
        key_params = {
            "RandomForestClassifier": [
                "n_estimators",
                "max_depth",
                "min_samples_split",
            ],
            "RandomForestRegressor": ["n_estimators", "max_depth", "min_samples_split"],
            "XGBClassifier": [
                "n_estimators",
                "max_depth",
                "learning_rate",
                "subsample",
            ],
            "XGBRegressor": ["n_estimators", "max_depth", "learning_rate", "subsample"],
            "LGBMRegressor": ["n_estimators", "num_leaves", "learning_rate"],
            "CatBoostRegressor": ["iterations", "depth", "learning_rate"],
            "LogisticRegression": ["C", "penalty"],
            "Ridge": ["alpha"],
            "Lasso": ["alpha"],
            "ElasticNet": ["alpha", "l1_ratio"],
            "SVC": ["C", "kernel", "gamma"],
            "SVR": ["C", "kernel", "gamma", "epsilon"],
            "GradientBoostingClassifier": [
                "n_estimators",
                "max_depth",
                "learning_rate",
            ],
            "GradientBoostingRegressor": ["n_estimators", "max_depth", "learning_rate"],
        }

        if model_name in key_params:
            compact_space = {
                param: full_space[param]
                for param in key_params[model_name]
                if param in full_space
            }
            return compact_space

        # Return top 3 parameters for unknown models
        return dict(list(full_space.items())[:3])

    def get_extended_space(self, model_name: str) -> Dict[str, Any]:
        """
        Get an extended search space with wider ranges for thorough optimization.

        Args:
            model_name: Name of the model

        Returns:
            Extended search space
        """
        base_space = self.get_search_space(model_name)
        extended_space = base_space.copy()

        # Extend numeric ranges
        for param, config in extended_space.items():
            if config["type"] in ["int", "float"]:
                # Extend range by 50%
                range_size = config["high"] - config["low"]
                config["low"] = max(0, config["low"] - range_size * 0.25)
                config["high"] = config["high"] + range_size * 0.25

        return extended_space

    def merge_spaces(self, *spaces: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple search spaces.

        Args:
            *spaces: Search spaces to merge

        Returns:
            Merged search space
        """
        merged = {}

        for space in spaces:
            for param, config in space.items():
                if param not in merged:
                    merged[param] = config
                else:
                    # Merge configs (take union of choices, min of lows, max of highs)
                    if config["type"] == "categorical":
                        existing_choices = set(merged[param]["choices"])
                        new_choices = set(config["choices"])
                        merged[param]["choices"] = list(
                            existing_choices.union(new_choices)
                        )
                    elif config["type"] in ["int", "float"]:
                        merged[param]["low"] = min(merged[param]["low"], config["low"])
                        merged[param]["high"] = max(
                            merged[param]["high"], config["high"]
                        )

        return merged


# Convenience functions
def get_default_search_space(model_name: str) -> Dict[str, Any]:
    """Get default search space for a model."""
    builder = SearchSpaceBuilder()
    return builder.get_search_space(model_name)


def get_compact_search_space(model_name: str) -> Dict[str, Any]:
    """Get compact search space for quick optimization."""
    builder = SearchSpaceBuilder()
    return builder.get_compact_space(model_name)


# Export public API
__all__ = ["SearchSpaceBuilder", "get_default_search_space", "get_compact_search_space"]
