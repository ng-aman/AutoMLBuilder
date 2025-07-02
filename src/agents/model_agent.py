# src/agents/model_agent.py
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from datetime import datetime
from langchain.schema import BaseMessage, HumanMessage
from langchain.tools import Tool
from src.agents.base_agent import BaseAgent
from src.core.config import AGENT_PROMPTS, ML_CONFIG
from src.core.state import (
    ConversationState,
    AgentType,
    ModelResult,
    add_message,
    add_warning,
    update_state_timestamp,
)
from src.ml.models.classifier import ModelTrainer as ClassifierTrainer
from src.ml.models.regressor import ModelTrainer as RegressorTrainer
from src.api.models.experiment import Experiment
from src.api.dependencies.database import SessionLocal
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelAgent(BaseAgent):
    """
    Agent responsible for training and evaluating machine learning models.
    """

    def __init__(self):
        # Initialize tools
        tools = [
            Tool(
                name="train_model",
                func=self._train_model_wrapper,
                description="Train a machine learning model",
            ),
            Tool(
                name="evaluate_model",
                func=self._evaluate_model_wrapper,
                description="Evaluate a trained model",
            ),
            Tool(
                name="compare_models",
                func=self._compare_models_wrapper,
                description="Compare multiple trained models",
            ),
        ]

        super().__init__(
            name="ModelAgent",
            agent_type=AgentType.MODEL,
            tools=tools,
            system_prompt=AGENT_PROMPTS["model"],
        )

        # Initialize MLflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment("automl_experiments")

    def _get_default_prompt(self) -> str:
        """Get default system prompt"""
        return AGENT_PROMPTS["model"]

    async def _execute(
        self, state: ConversationState, messages: List[BaseMessage]
    ) -> Tuple[ConversationState, str]:
        """Execute model training workflow"""

        # Check prerequisites
        if not state["applied_transformations"]:
            return state, "Data must be preprocessed before model training."

        if not state["problem_type"]:
            return state, "Problem type must be identified before model training."

        # Record initial thought
        state = self._record_thought(
            state, f"Starting model training for {state['problem_type']} problem"
        )

        try:
            # Get model selection
            models_to_train = await self._select_models(state, messages)

            # Train models
            training_results = []
            for model_name in models_to_train:
                result = await self._train_single_model(state, model_name)
                training_results.append(result)

                # Update state
                state["models_trained"].append(result["summary"])
                state["model_results"][model_name] = result

            # Compare and select best model
            best_model = self._select_best_model(training_results)
            state["best_model"] = best_model

            # Create summary
            summary = self._create_training_summary(training_results, best_model)

            # Update state
            state = add_message(
                state,
                "assistant",
                summary,
                metadata={
                    "agent": self.agent_type.value,
                    "models_trained": len(training_results),
                    "best_model": best_model["name"],
                },
            )

            # Record completion
            state = self._record_thought(
                state,
                f"Model training completed. Best model: {best_model['name']}",
                decision=f"Accuracy: {best_model['metrics']['accuracy']:.3f}",
            )

            return state, summary

        except Exception as e:
            logger.error("Model training failed", error=str(e))
            error_message = f"Model training encountered an error: {str(e)}"
            state = add_message(state, "assistant", error_message)
            raise

    async def _select_models(
        self, state: ConversationState, messages: List[BaseMessage]
    ) -> List[str]:
        """Select models to train based on problem type and data"""
        problem_type = state["problem_type"]

        # Get default models for problem type
        if problem_type == "classification":
            default_models = ML_CONFIG["classification"]["models"]
        else:
            default_models = ML_CONFIG["regression"]["models"]

        # Check if user specified models
        user_message = messages[-1].content if messages else ""

        # Create prompt for model selection
        prompt = f"""
Based on the problem type ({problem_type}) and dataset characteristics, 
select the most appropriate models to train.

Dataset info:
- Rows: {state['dataset_info']['rows']}
- Features: {state['dataset_info']['columns']}
- Problem type: {problem_type}

Available models: {default_models}
User request: {user_message}

Select 3-5 models that would work well for this dataset.
List them in order of expected performance.
"""

        # Get LLM response
        response = await self._call_llm(
            messages + [HumanMessage(content=prompt)],
            session_id=state["session_id"],
            user_id=state["user_id"],
        )

        # Parse model selection (for now, use defaults)
        # In production, parse LLM response
        selected_models = default_models[:4]  # Select top 4 models

        logger.info(f"Selected models for training: {selected_models}")
        return selected_models

    async def _train_single_model(
        self, state: ConversationState, model_name: str
    ) -> Dict[str, Any]:
        """Train a single model"""
        logger.info(f"Training {model_name}...")

        # Start MLflow run
        with mlflow.start_run(
            run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ):
            # Log parameters
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("problem_type", state["problem_type"])
            mlflow.log_param("session_id", state["session_id"])

            # Get trainer
            if state["problem_type"] == "classification":
                trainer = ClassifierTrainer()
            else:
                trainer = RegressorTrainer()

            # Load preprocessed data (mock for now)
            X_train, X_test, y_train, y_test = self._load_preprocessed_data(state)

            # Train model
            model, metrics = trainer.train(
                X_train, y_train, X_test, y_test, model_name=model_name
            )

            # Log metrics
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)

            # Log model
            mlflow.sklearn.log_model(model, model_name)

            # Save to database
            run_id = mlflow.active_run().info.run_id
            self._save_experiment_to_db(state, model_name, metrics, run_id)

            # Track event
            await event_tracker.track_model_training(
                session_id=state["session_id"],
                model_name=model_name,
                parameters=model.get_params() if hasattr(model, "get_params") else {},
                metrics=metrics,
                duration_ms=(datetime.now() - datetime.now()).total_seconds() * 1000,
                mlflow_run_id=run_id,
                user_id=state["user_id"],
            )

            return {
                "name": model_name,
                "model": model,
                "metrics": metrics,
                "mlflow_run_id": run_id,
                "summary": {
                    "model": model_name,
                    "accuracy": metrics.get("accuracy", metrics.get("r2", 0)),
                },
            }

    def _load_preprocessed_data(self, state: ConversationState):
        """Load preprocessed data (mock implementation)"""
        # In production, load from processed data path
        # For now, create mock data
        n_samples = 1000
        n_features = 20

        X_train = np.random.randn(n_samples, n_features)
        X_test = np.random.randn(n_samples // 4, n_features)

        if state["problem_type"] == "classification":
            y_train = np.random.randint(0, 2, n_samples)
            y_test = np.random.randint(0, 2, n_samples // 4)
        else:
            y_train = np.random.randn(n_samples)
            y_test = np.random.randn(n_samples // 4)

        return X_train, X_test, y_train, y_test

    def _save_experiment_to_db(
        self,
        state: ConversationState,
        model_name: str,
        metrics: Dict[str, float],
        mlflow_run_id: str,
    ):
        """Save experiment to database"""
        try:
            db = SessionLocal()

            experiment = Experiment(
                session_id=state["session_id"],
                mlflow_run_id=mlflow_run_id,
                status="completed",
                results={
                    "model": model_name,
                    "metrics": metrics,
                    "problem_type": state["problem_type"],
                },
            )

            db.add(experiment)
            db.commit()
            db.close()

        except Exception as e:
            logger.error("Failed to save experiment", error=str(e))

    def _select_best_model(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select best model based on primary metric"""
        if not results:
            return None

        # Sort by primary metric (accuracy for classification, r2 for regression)
        primary_metric = "accuracy" if "accuracy" in results[0]["metrics"] else "r2"

        best_model = max(results, key=lambda x: x["metrics"].get(primary_metric, 0))

        return best_model

    def _create_training_summary(
        self, results: List[Dict[str, Any]], best_model: Dict[str, Any]
    ) -> str:
        """Create training summary"""
        summary = """
🤖 **Model Training Complete**

**Models Trained:**
"""

        # Add results for each model
        for result in results:
            model_name = result["name"]
            metrics = result["metrics"]

            summary += f"\n**{model_name}**\n"
            for metric, value in metrics.items():
                summary += f"- {metric.title()}: {value:.3f}\n"

        # Highlight best model
        if best_model:
            summary += f"""
🏆 **Best Model: {best_model['name']}**
- Primary Metric: {best_model['metrics'].get('accuracy', best_model['metrics'].get('r2', 0)):.3f}
- MLflow Run ID: {best_model['mlflow_run_id']}

✅ Model training completed successfully! The best model has been selected for deployment.
"""

        return summary

    # Tool wrapper methods
    def _train_model_wrapper(self, model_name: str) -> str:
        """Wrapper for train_model tool"""
        return f"Training {model_name} model..."

    def _evaluate_model_wrapper(self, model_name: str) -> str:
        """Wrapper for evaluate_model tool"""
        return f"Evaluating {model_name} model..."

    def _compare_models_wrapper(self, models: List[str]) -> str:
        """Wrapper for compare_models tool"""
        return f"Comparing models: {', '.join(models)}..."
