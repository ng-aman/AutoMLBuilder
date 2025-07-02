# src/agents/optimization_agent.py
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
import optuna
import mlflow
from datetime import datetime
from langchain.schema import BaseMessage, HumanMessage
from langchain.tools import Tool
from src.agents.base_agent import BaseAgent
from src.core.config import AGENT_PROMPTS, settings
from src.core.state import (
    ConversationState,
    AgentType,
    add_message,
    update_state_timestamp,
)
from src.ml.optimization.optuna_optimizer import OptunaOptimizer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OptimizationAgent(BaseAgent):
    """
    Agent responsible for hyperparameter optimization using Optuna.
    """

    def __init__(self):
        # Initialize tools
        tools = [
            Tool(
                name="create_study",
                func=self._create_study_wrapper,
                description="Create an Optuna study for optimization",
            ),
            Tool(
                name="optimize_hyperparameters",
                func=self._optimize_wrapper,
                description="Run hyperparameter optimization",
            ),
            Tool(
                name="analyze_optimization",
                func=self._analyze_optimization_wrapper,
                description="Analyze optimization results",
            ),
        ]

        super().__init__(
            name="OptimizationAgent",
            agent_type=AgentType.OPTIMIZATION,
            tools=tools,
            system_prompt=AGENT_PROMPTS["optimization"],
        )

        self.optimizer = None
        self.study = None

    def _get_default_prompt(self) -> str:
        """Get default system prompt"""
        return AGENT_PROMPTS["optimization"]

    async def _execute(
        self, state: ConversationState, messages: List[BaseMessage]
    ) -> Tuple[ConversationState, str]:
        """Execute hyperparameter optimization workflow"""

        # Check prerequisites
        if not state["models_trained"]:
            return state, "Models must be trained before optimization."

        if not state["best_model"]:
            return state, "No best model identified for optimization."

        # Record initial thought
        state = self._record_thought(
            state,
            f"Starting hyperparameter optimization for {state['best_model']['model']}",
        )

        try:
            # Get optimization configuration
            config = await self._get_optimization_config(state, messages)

            # Create Optuna study
            study_name = f"automl_{state['session_id'][:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.study = await self._create_study(study_name, config)
            state["optimization_study_name"] = study_name

            # Run optimization
            optimization_results = await self._run_optimization(state, config)

            # Analyze results
            analysis = self._analyze_results(self.study)

            # Update state with best parameters
            state["best_params"] = self.study.best_params
            state["optimization_history"] = optimization_results

            # Create summary
            summary = self._create_optimization_summary(
                self.study, analysis, optimization_results
            )

            # Update state
            state = add_message(
                state,
                "assistant",
                summary,
                metadata={
                    "agent": self.agent_type.value,
                    "study_name": study_name,
                    "n_trials": len(optimization_results),
                    "best_value": self.study.best_value,
                },
            )

            # Record completion
            state = self._record_thought(
                state,
                "Optimization completed successfully",
                decision=f"Best score: {self.study.best_value:.4f}",
            )

            return state, summary

        except Exception as e:
            logger.error("Optimization failed", error=str(e))
            error_message = f"Optimization encountered an error: {str(e)}"
            state = add_message(state, "assistant", error_message)
            raise

    async def _get_optimization_config(
        self, state: ConversationState, messages: List[BaseMessage]
    ) -> Dict[str, Any]:
        """Get optimization configuration"""

        # Create prompt for configuration
        prompt = f"""
Based on the best model ({state['best_model']['model']}) and the problem type ({state['problem_type']}),
determine the optimization configuration.

Current model performance: {state['best_model'].get('accuracy', state['best_model'].get('r2', 0)):.3f}

Suggest:
1. Number of trials (10-200)
2. Optimization direction (maximize/minimize)
3. Key hyperparameters to optimize
4. Timeout in seconds (300-3600)

Consider the dataset size and current performance.
"""

        # Get LLM response
        response = await self._call_llm(
            messages + [HumanMessage(content=prompt)],
            session_id=state["session_id"],
            user_id=state["user_id"],
        )

        # Default configuration
        config = {
            "n_trials": 50,
            "direction": "maximize",
            "timeout": 1800,  # 30 minutes
            "n_jobs": 1,
            "model_name": state["best_model"]["model"],
            "metric": "accuracy" if state["problem_type"] == "classification" else "r2",
        }

        # TODO: Parse LLM response to update config

        logger.info(f"Optimization config: {config}")
        return config

    async def _create_study(
        self, study_name: str, config: Dict[str, Any]
    ) -> optuna.Study:
        """Create Optuna study"""
        logger.info(f"Creating Optuna study: {study_name}")

        # Create study
        study = optuna.create_study(
            study_name=study_name,
            direction=config["direction"],
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(),
        )

        # Log to MLflow
        mlflow.set_experiment(f"optuna_{study_name}")

        return study

    async def _run_optimization(
        self, state: ConversationState, config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Run hyperparameter optimization"""
        logger.info(f"Starting optimization with {config['n_trials']} trials")

        # Initialize optimizer
        self.optimizer = OptunaOptimizer(
            model_name=config["model_name"], problem_type=state["problem_type"]
        )

        # Load data (mock for now)
        X_train, X_test, y_train, y_test = self._load_data(state)

        # Create objective function
        def objective(trial):
            # Get hyperparameters
            params = self.optimizer.get_search_space(trial)

            # Train model with params
            score = self.optimizer.train_and_evaluate(
                trial, X_train, y_train, X_test, y_test, params, config["metric"]
            )

            # Track event
            asyncio.create_task(
                event_tracker.track_agent_action(
                    session_id=state["session_id"],
                    agent_name=self.name,
                    action="optimization_trial",
                    output_data={
                        "trial": trial.number,
                        "params": params,
                        "score": score,
                    },
                    user_id=state["user_id"],
                )
            )

            return score

        # Run optimization
        import asyncio

        self.study.optimize(
            objective,
            n_trials=config["n_trials"],
            timeout=config["timeout"],
            n_jobs=config["n_jobs"],
        )

        # Collect results
        results = []
        for trial in self.study.trials:
            results.append(
                {
                    "trial": trial.number,
                    "params": trial.params,
                    "value": trial.value,
                    "state": trial.state.name,
                }
            )

        logger.info(f"Optimization completed. Best value: {self.study.best_value}")
        return results

    def _load_data(self, state: ConversationState):
        """Load preprocessed data (mock implementation)"""
        # Same as model agent - in production, load from processed path
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

    def _analyze_results(self, study: optuna.Study) -> Dict[str, Any]:
        """Analyze optimization results"""
        analysis = {
            "n_trials": len(study.trials),
            "n_complete": len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            ),
            "n_pruned": len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
            ),
            "best_value": study.best_value,
            "best_params": study.best_params,
            "best_trial": study.best_trial.number,
        }

        # Parameter importance
        try:
            importance = optuna.importance.get_param_importances(study)
            analysis["param_importance"] = importance
        except:
            analysis["param_importance"] = {}

        return analysis

    def _create_optimization_summary(
        self,
        study: optuna.Study,
        analysis: Dict[str, Any],
        results: List[Dict[str, Any]],
    ) -> str:
        """Create optimization summary"""
        summary = f"""
🎯 **Hyperparameter Optimization Complete**

**Study Summary:**
- Total Trials: {analysis['n_trials']}
- Completed: {analysis['n_complete']}
- Pruned: {analysis['n_pruned']}

**Best Results:**
- Best Score: {analysis['best_value']:.4f}
- Best Trial: #{analysis['best_trial']}

**Optimized Parameters:**
"""

        # Add best parameters
        for param, value in analysis["best_params"].items():
            summary += f"- {param}: {value}\n"

        # Add parameter importance if available
        if analysis.get("param_importance"):
            summary += "\n**Parameter Importance:**\n"
            sorted_importance = sorted(
                analysis["param_importance"].items(), key=lambda x: x[1], reverse=True
            )
            for param, importance in sorted_importance[:5]:
                summary += f"- {param}: {importance:.3f}\n"

        # Add improvement
        if results:
            initial_score = results[0]["value"]
            improvement = (
                (analysis["best_value"] - initial_score) / initial_score
            ) * 100
            summary += f"\n**Improvement: {improvement:+.1f}%**"

        summary += """

✅ Hyperparameter optimization completed successfully! 
The optimized parameters have been saved and can be used for final model training.
"""

        return summary

    # Tool wrapper methods
    def _create_study_wrapper(self, study_name: str) -> str:
        """Wrapper for create_study tool"""
        return f"Creating Optuna study: {study_name}"

    def _optimize_wrapper(self, n_trials: int) -> str:
        """Wrapper for optimize tool"""
        return f"Running optimization with {n_trials} trials..."

    def _analyze_optimization_wrapper(self, study_name: str) -> str:
        """Wrapper for analyze tool"""
        return f"Analyzing optimization results for {study_name}..."
