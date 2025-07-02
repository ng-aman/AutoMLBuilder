# src/workflows/nodes.py
from typing import Dict, Any
import asyncio
from src.core.state import (
    ConversationState,
    WorkflowStatus,
    AgentType,
    add_message,
    add_error,
    update_state_timestamp,
)
from src.agents.supervisor import SupervisorAgent
from src.agents.analysis_agent import AnalysisAgent
from src.agents.preprocessing_agent import PreprocessingAgent
from src.core.events import event_tracker
from src.utils.logger import get_logger
from src.utils.exceptions import AgentError

logger = get_logger(__name__)

# Initialize agents (these could be singletons)
supervisor_agent = SupervisorAgent()
analysis_agent = AnalysisAgent()
preprocessing_agent = PreprocessingAgent()


async def supervisor_node(state: ConversationState) -> ConversationState:
    """
    Supervisor node that orchestrates the workflow.
    """
    try:
        logger.info("Executing supervisor node", session_id=state["session_id"])

        # Track node execution
        await event_tracker.track_workflow_update(
            session_id=state["session_id"],
            status="supervisor_active",
            current_agent="supervisor",
            user_id=state["user_id"],
        )

        # Get last user message if any
        user_message = None
        for msg in reversed(state["messages"]):
            if msg["role"] == "user":
                user_message = msg["content"]
                break

        # Execute supervisor agent
        state = await supervisor_agent(state, user_message)

        return state

    except Exception as e:
        logger.error("Supervisor node error", error=str(e))
        state = add_error(
            state,
            error=str(e),
            agent=AgentType.SUPERVISOR,
            details={"node": "supervisor"},
        )
        state["status"] = WorkflowStatus.FAILED
        return state


async def analysis_node(state: ConversationState) -> ConversationState:
    """
    Analysis node for data exploration and problem identification.
    """
    try:
        logger.info("Executing analysis node", session_id=state["session_id"])

        # Check prerequisites
        if not state["dataset_id"]:
            state = add_message(
                state,
                "system",
                "Cannot perform analysis without a dataset. Please upload a dataset first.",
            )
            return state

        # Track node execution
        await event_tracker.track_workflow_update(
            session_id=state["session_id"],
            status="analyzing",
            current_agent="analysis",
            user_id=state["user_id"],
        )

        # Execute analysis agent
        state = await analysis_agent(state)

        return state

    except Exception as e:
        logger.error("Analysis node error", error=str(e))
        state = add_error(
            state, error=str(e), agent=AgentType.ANALYSIS, details={"node": "analysis"}
        )
        return state


async def preprocessing_node(state: ConversationState) -> ConversationState:
    """
    Preprocessing node for data cleaning and transformation.
    """
    try:
        logger.info("Executing preprocessing node", session_id=state["session_id"])

        # Check prerequisites
        if not state["dataset_info"]:
            state = add_message(
                state,
                "system",
                "Cannot preprocess without dataset analysis. Running analysis first.",
            )
            state["current_agent"] = AgentType.ANALYSIS
            return state

        # Track node execution
        await event_tracker.track_workflow_update(
            session_id=state["session_id"],
            status="preprocessing",
            current_agent="preprocessing",
            user_id=state["user_id"],
        )

        # Execute preprocessing agent
        state = await preprocessing_agent(state)

        return state

    except Exception as e:
        logger.error("Preprocessing node error", error=str(e))
        state = add_error(
            state,
            error=str(e),
            agent=AgentType.PREPROCESSING,
            details={"node": "preprocessing"},
        )
        return state


async def model_node(state: ConversationState) -> ConversationState:
    """
    Model training node.
    """
    try:
        logger.info("Executing model node", session_id=state["session_id"])

        # Check prerequisites
        if not state["applied_transformations"]:
            state = add_message(
                state,
                "system",
                "Cannot train models without preprocessed data. Running preprocessing first.",
            )
            state["current_agent"] = AgentType.PREPROCESSING
            return state

        # Track node execution
        await event_tracker.track_workflow_update(
            session_id=state["session_id"],
            status="training",
            current_agent="model",
            user_id=state["user_id"],
        )

        # Placeholder for model agent (to be implemented)
        state = add_message(
            state,
            "assistant",
            "Model training is being implemented. This would train multiple ML models and compare their performance.",
        )

        # Simulate some model results
        state["models_trained"] = [
            {"model": "RandomForest", "accuracy": 0.85},
            {"model": "XGBoost", "accuracy": 0.87},
            {"model": "LogisticRegression", "accuracy": 0.82},
        ]
        state["best_model"] = {"model": "XGBoost", "accuracy": 0.87}

        return state

    except Exception as e:
        logger.error("Model node error", error=str(e))
        state = add_error(
            state, error=str(e), agent=AgentType.MODEL, details={"node": "model"}
        )
        return state


async def optimization_node(state: ConversationState) -> ConversationState:
    """
    Hyperparameter optimization node.
    """
    try:
        logger.info("Executing optimization node", session_id=state["session_id"])

        # Check prerequisites
        if not state["models_trained"]:
            state = add_message(
                state,
                "system",
                "Cannot optimize without trained models. Running model training first.",
            )
            state["current_agent"] = AgentType.MODEL
            return state

        # Track node execution
        await event_tracker.track_workflow_update(
            session_id=state["session_id"],
            status="optimizing",
            current_agent="optimization",
            user_id=state["user_id"],
        )

        # Placeholder for optimization agent
        state = add_message(
            state,
            "assistant",
            "Hyperparameter optimization is being implemented. This would use Optuna to find optimal parameters.",
        )

        # Simulate optimization results
        state["best_params"] = {
            "n_estimators": 200,
            "max_depth": 10,
            "learning_rate": 0.1,
        }
        state["optimization_history"] = [
            {"trial": 1, "score": 0.85},
            {"trial": 2, "score": 0.88},
            {"trial": 3, "score": 0.89},
        ]

        return state

    except Exception as e:
        logger.error("Optimization node error", error=str(e))
        state = add_error(
            state,
            error=str(e),
            agent=AgentType.OPTIMIZATION,
            details={"node": "optimization"},
        )
        return state


async def human_approval_node(state: ConversationState) -> ConversationState:
    """
    Human approval node for interactive mode.
    """
    try:
        logger.info("Executing human approval node", session_id=state["session_id"])

        if not state.get("pending_approval"):
            logger.warning("No pending approval found")
            return state

        # Track node execution
        await event_tracker.track_workflow_update(
            session_id=state["session_id"],
            status=WorkflowStatus.WAITING_FOR_APPROVAL,
            current_agent="human_approval",
            user_id=state["user_id"],
        )

        # In a real implementation, this would wait for user input
        # For now, we'll just return the state as-is
        # The actual approval handling happens in the API endpoint

        return state

    except Exception as e:
        logger.error("Human approval node error", error=str(e))
        state = add_error(state, error=str(e), details={"node": "human_approval"})
        return state


async def error_handler_node(state: ConversationState) -> ConversationState:
    """
    Error handler node for recovering from errors.
    """
    try:
        logger.info("Executing error handler node", session_id=state["session_id"])

        # Get recent errors
        recent_errors = state["errors"][-3:] if state["errors"] else []

        if len(recent_errors) >= 3:
            # Too many errors, fail the workflow
            state["status"] = WorkflowStatus.FAILED
            state = add_message(
                state,
                "system",
                "Workflow failed due to multiple errors. Please check the logs and try again.",
            )
        else:
            # Try to recover
            state = add_message(
                state,
                "system",
                f"Attempting to recover from error: {recent_errors[-1]['error'] if recent_errors else 'Unknown error'}",
            )

            # Reset to supervisor
            state["current_agent"] = AgentType.SUPERVISOR
            state["status"] = WorkflowStatus.RUNNING

        return state

    except Exception as e:
        logger.error("Error handler node error", error=str(e))
        # If error handler fails, just fail the workflow
        state["status"] = WorkflowStatus.FAILED
        return state


# Sync wrappers for LangGraph compatibility
def supervisor_node_sync(state: ConversationState) -> ConversationState:
    """Sync wrapper for supervisor node"""
    return asyncio.run(supervisor_node(state))


def analysis_node_sync(state: ConversationState) -> ConversationState:
    """Sync wrapper for analysis node"""
    return asyncio.run(analysis_node(state))


def preprocessing_node_sync(state: ConversationState) -> ConversationState:
    """Sync wrapper for preprocessing node"""
    return asyncio.run(preprocessing_node(state))


def model_node_sync(state: ConversationState) -> ConversationState:
    """Sync wrapper for model node"""
    return asyncio.run(model_node(state))


def optimization_node_sync(state: ConversationState) -> ConversationState:
    """Sync wrapper for optimization node"""
    return asyncio.run(optimization_node(state))


def human_approval_node_sync(state: ConversationState) -> ConversationState:
    """Sync wrapper for human approval node"""
    return asyncio.run(human_approval_node(state))


def error_handler_node_sync(state: ConversationState) -> ConversationState:
    """Sync wrapper for error handler node"""
    return asyncio.run(error_handler_node(state))


# Export sync versions for LangGraph
supervisor_node = supervisor_node_sync
analysis_node = analysis_node_sync
preprocessing_node = preprocessing_node_sync
model_node = model_node_sync
optimization_node = optimization_node_sync
human_approval_node = human_approval_node_sync
error_handler_node = error_handler_node_sync
