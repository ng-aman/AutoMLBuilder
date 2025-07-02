# src/workflows/edges.py
from typing import Literal
from src.core.state import (
    ConversationState,
    WorkflowStatus,
    AgentType,
    is_workflow_complete,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def should_continue(state: ConversationState) -> bool:
    """
    Determine if the workflow should continue or end.
    """
    # Check if workflow is complete
    if is_workflow_complete(state):
        return False

    # Check if there are too many errors
    if len(state["errors"]) > 5:
        return False

    # Check if max iterations reached
    if len(state["messages"]) > 100:
        logger.warning("Max iterations reached", session_id=state["session_id"])
        return False

    return True


def route_supervisor(
    state: ConversationState,
) -> Literal[
    "analysis",
    "preprocessing",
    "model",
    "optimization",
    "human_approval",
    "error",
    "end",
]:
    """
    Route from supervisor to next node.
    """
    # Check for pending approval
    if state.get("pending_approval"):
        return "human_approval"

    # Check for errors
    if state["errors"] and state["status"] == WorkflowStatus.FAILED:
        return "error"

    # Check if workflow is complete
    if state["status"] == WorkflowStatus.COMPLETED:
        return "end"

    # Route based on current agent
    current_agent = state.get("current_agent")

    if current_agent == AgentType.ANALYSIS:
        return "analysis"
    elif current_agent == AgentType.PREPROCESSING:
        return "preprocessing"
    elif current_agent == AgentType.MODEL:
        return "model"
    elif current_agent == AgentType.OPTIMIZATION:
        return "optimization"
    else:
        # Default routing logic based on workflow state
        if not state["dataset_id"]:
            # No dataset, wait for user input
            return "end"
        elif not state["dataset_info"]:
            # Dataset exists but not analyzed
            return "analysis"
        elif not state["applied_transformations"]:
            # Dataset analyzed but not preprocessed
            return "preprocessing"
        elif not state["models_trained"]:
            # Data preprocessed but no models trained
            return "model"
        elif not state["best_params"]:
            # Models trained but not optimized
            return "optimization"
        else:
            # Everything done
            return "end"


def route_after_analysis(
    state: ConversationState,
) -> Literal["supervisor", "preprocessing", "human_approval", "error"]:
    """
    Route after analysis agent completes.
    """
    # Check for errors
    if state["errors"] and state["errors"][-1].get("agent") == AgentType.ANALYSIS.value:
        return "error"

    # Check for pending approval
    if state.get("pending_approval"):
        return "human_approval"

    # Check if we should go directly to preprocessing
    if state["dataset_info"] and state["target_variable"] and state["problem_type"]:
        # Analysis complete, can proceed to preprocessing
        if state["mode"] == "auto":
            return "preprocessing"

    # Default: go back to supervisor
    return "supervisor"


def route_after_preprocessing(
    state: ConversationState,
) -> Literal["supervisor", "model", "human_approval", "error"]:
    """
    Route after preprocessing agent completes.
    """
    # Check for errors
    if (
        state["errors"]
        and state["errors"][-1].get("agent") == AgentType.PREPROCESSING.value
    ):
        return "error"

    # Check for pending approval
    if state.get("pending_approval"):
        return "human_approval"

    # Check if preprocessing is complete
    if state["applied_transformations"]:
        # Preprocessing complete, can proceed to model training
        if state["mode"] == "auto":
            return "model"

    # Default: go back to supervisor
    return "supervisor"


def route_after_model(
    state: ConversationState,
) -> Literal["supervisor", "optimization", "human_approval", "error", "end"]:
    """
    Route after model agent completes.
    """
    # Check for errors
    if state["errors"] and state["errors"][-1].get("agent") == AgentType.MODEL.value:
        return "error"

    # Check for pending approval
    if state.get("pending_approval"):
        return "human_approval"

    # Check if models are trained
    if state["models_trained"] and state["best_model"]:
        # Models trained successfully
        if state["mode"] == "auto":
            # In auto mode, proceed to optimization
            return "optimization"
        else:
            # In interactive mode, check if user wants optimization
            # For now, go back to supervisor
            return "supervisor"

    # If no models trained, this might be the end
    if state["status"] == WorkflowStatus.COMPLETED:
        return "end"

    # Default: go back to supervisor
    return "supervisor"


def route_after_optimization(
    state: ConversationState,
) -> Literal["supervisor", "human_approval", "error", "end"]:
    """
    Route after optimization agent completes.
    """
    # Check for errors
    if (
        state["errors"]
        and state["errors"][-1].get("agent") == AgentType.OPTIMIZATION.value
    ):
        return "error"

    # Check for pending approval
    if state.get("pending_approval"):
        return "human_approval"

    # Check if optimization is complete
    if state["best_params"] and state["optimization_history"]:
        # Optimization complete, workflow is done
        state["status"] = WorkflowStatus.COMPLETED
        return "end"

    # Default: go back to supervisor
    return "supervisor"


def route_after_approval(state: ConversationState) -> Literal["supervisor"]:
    """
    Route after human approval. Always goes back to supervisor.
    """
    return "supervisor"


def route_after_error(state: ConversationState) -> Literal["supervisor", "end"]:
    """
    Route after error handler.
    """
    # Check if too many errors
    if len(state["errors"]) > 3:
        state["status"] = WorkflowStatus.FAILED
        return "end"

    # Try to recover by going back to supervisor
    return "supervisor"


# Helper functions for specific routing decisions


def needs_human_approval(state: ConversationState) -> bool:
    """
    Check if human approval is needed.
    """
    return state["mode"] == "interactive" and state.get("pending_approval") is not None


def can_proceed_to_next_stage(state: ConversationState) -> bool:
    """
    Check if workflow can proceed to next stage.
    """
    # Must not have pending approval
    if state.get("pending_approval"):
        return False

    # Must not be in error state
    if state["status"] == WorkflowStatus.FAILED:
        return False

    # Must not be waiting for approval
    if state["status"] == WorkflowStatus.WAITING_FOR_APPROVAL:
        return False

    return True


def get_next_agent(state: ConversationState) -> AgentType:
    """
    Determine the next agent based on workflow state.
    """
    if not state["dataset_info"]:
        return AgentType.ANALYSIS
    elif not state["applied_transformations"]:
        return AgentType.PREPROCESSING
    elif not state["models_trained"]:
        return AgentType.MODEL
    elif not state["best_params"]:
        return AgentType.OPTIMIZATION
    else:
        return AgentType.SUPERVISOR  # Default
