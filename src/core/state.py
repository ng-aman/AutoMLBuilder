# src/core/state.py
from typing import Dict, List, Optional, Any, Literal, TypedDict
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
import uuid


class WorkflowMode(str, Enum):
    """Workflow execution modes"""

    AUTO = "auto"
    INTERACTIVE = "interactive"


class ProblemType(str, Enum):
    """ML problem types"""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    UNKNOWN = "unknown"


class WorkflowStatus(str, Enum):
    """Workflow status"""

    IDLE = "idle"
    RUNNING = "running"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentType(str, Enum):
    """Agent types"""

    SUPERVISOR = "supervisor"
    ANALYSIS = "analysis"
    PREPROCESSING = "preprocessing"
    MODEL = "model"
    OPTIMIZATION = "optimization"


# TypedDict for LangGraph state
class ConversationState(TypedDict):
    """Main state for LangGraph workflow"""

    # Session information
    session_id: str
    user_id: str
    created_at: str
    updated_at: str

    # Workflow configuration
    mode: WorkflowMode
    status: WorkflowStatus
    current_agent: Optional[AgentType]
    debug_enabled: bool

    # Dataset information
    dataset_id: Optional[str]
    dataset_info: Optional[Dict[str, Any]]
    target_variable: Optional[str]
    problem_type: Optional[ProblemType]

    # Processing state
    preprocessing_steps: List[Dict[str, Any]]
    applied_transformations: List[Dict[str, Any]]
    feature_importance: Optional[Dict[str, float]]

    # Model training state
    models_trained: List[Dict[str, Any]]
    model_results: Dict[str, Dict[str, Any]]
    best_model: Optional[Dict[str, Any]]

    # Optimization state
    optimization_history: List[Dict[str, Any]]
    best_params: Optional[Dict[str, Any]]
    optimization_study_name: Optional[str]

    # Conversation history
    messages: List[Dict[str, Any]]
    agent_thoughts: List[Dict[str, Any]]

    # Human-in-the-loop
    pending_approval: Optional[Dict[str, Any]]
    user_decisions: List[Dict[str, Any]]

    # Error tracking
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]


# Pydantic models for validation
class Message(BaseModel):
    """Chat message model"""

    role: Literal["user", "assistant", "system", "agent"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None


class AgentThought(BaseModel):
    """Agent thought/reasoning model"""

    agent: AgentType
    thought: str
    decision: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DatasetInfo(BaseModel):
    """Dataset information model"""

    dataset_id: str
    filename: str
    file_path: str
    file_size: int
    rows: int
    columns: int
    column_names: List[str]
    column_types: Dict[str, str]
    missing_values: Dict[str, int]
    statistics: Optional[Dict[str, Any]] = None


class PreprocessingStep(BaseModel):
    """Preprocessing step model"""

    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    parameters: Dict[str, Any]
    applied: bool = False
    timestamp: Optional[datetime] = None


class ModelResult(BaseModel):
    """Model training result"""

    model_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_name: str
    algorithm: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    training_time: float
    mlflow_run_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ApprovalRequest(BaseModel):
    """Human approval request"""

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent: AgentType
    action: str
    description: str
    options: List[str]
    default_option: Optional[str] = None
    timeout_seconds: int = 600
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class UserDecision(BaseModel):
    """User decision for approval request"""

    request_id: str
    decision: str
    reason: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# State management functions
def create_initial_state(
    session_id: str,
    user_id: str,
    mode: WorkflowMode = WorkflowMode.AUTO,
    debug_enabled: bool = False,
) -> ConversationState:
    """Create initial conversation state"""
    now = datetime.utcnow().isoformat()

    return ConversationState(
        # Session information
        session_id=session_id,
        user_id=user_id,
        created_at=now,
        updated_at=now,
        # Workflow configuration
        mode=mode,
        status=WorkflowStatus.IDLE,
        current_agent=None,
        debug_enabled=debug_enabled,
        # Dataset information
        dataset_id=None,
        dataset_info=None,
        target_variable=None,
        problem_type=None,
        # Processing state
        preprocessing_steps=[],
        applied_transformations=[],
        feature_importance=None,
        # Model training state
        models_trained=[],
        model_results={},
        best_model=None,
        # Optimization state
        optimization_history=[],
        best_params=None,
        optimization_study_name=None,
        # Conversation history
        messages=[],
        agent_thoughts=[],
        # Human-in-the-loop
        pending_approval=None,
        user_decisions=[],
        # Error tracking
        errors=[],
        warnings=[],
    )


def update_state_timestamp(state: ConversationState) -> ConversationState:
    """Update the state timestamp"""
    state["updated_at"] = datetime.utcnow().isoformat()
    return state


def add_message(
    state: ConversationState,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> ConversationState:
    """Add a message to the conversation"""
    message = Message(role=role, content=content, metadata=metadata)
    state["messages"].append(message.dict())
    return update_state_timestamp(state)


def add_agent_thought(
    state: ConversationState,
    agent: AgentType,
    thought: str,
    decision: Optional[str] = None,
) -> ConversationState:
    """Add an agent thought to the state"""
    agent_thought = AgentThought(agent=agent, thought=thought, decision=decision)
    state["agent_thoughts"].append(agent_thought.dict())
    return update_state_timestamp(state)


def set_pending_approval(
    state: ConversationState,
    agent: AgentType,
    action: str,
    description: str,
    options: List[str],
    default_option: Optional[str] = None,
) -> ConversationState:
    """Set a pending approval request"""
    approval = ApprovalRequest(
        agent=agent,
        action=action,
        description=description,
        options=options,
        default_option=default_option,
    )
    state["pending_approval"] = approval.dict()
    state["status"] = WorkflowStatus.WAITING_FOR_APPROVAL
    return update_state_timestamp(state)


def resolve_approval(
    state: ConversationState, decision: str, reason: Optional[str] = None
) -> ConversationState:
    """Resolve a pending approval"""
    if state["pending_approval"]:
        user_decision = UserDecision(
            request_id=state["pending_approval"]["request_id"],
            decision=decision,
            reason=reason,
        )
        state["user_decisions"].append(user_decision.dict())
        state["pending_approval"] = None
        state["status"] = WorkflowStatus.RUNNING
    return update_state_timestamp(state)


def add_error(
    state: ConversationState,
    error: str,
    agent: Optional[AgentType] = None,
    details: Optional[Dict[str, Any]] = None,
) -> ConversationState:
    """Add an error to the state"""
    error_entry = {
        "error": error,
        "agent": agent.value if agent else None,
        "details": details or {},
        "timestamp": datetime.utcnow().isoformat(),
    }
    state["errors"].append(error_entry)
    return update_state_timestamp(state)


def add_warning(
    state: ConversationState,
    warning: str,
    agent: Optional[AgentType] = None,
    details: Optional[Dict[str, Any]] = None,
) -> ConversationState:
    """Add a warning to the state"""
    warning_entry = {
        "warning": warning,
        "agent": agent.value if agent else None,
        "details": details or {},
        "timestamp": datetime.utcnow().isoformat(),
    }
    state["warnings"].append(warning_entry)
    return update_state_timestamp(state)


def is_workflow_complete(state: ConversationState) -> bool:
    """Check if workflow is complete"""
    return state["status"] in [
        WorkflowStatus.COMPLETED,
        WorkflowStatus.FAILED,
        WorkflowStatus.CANCELLED,
    ]


def get_current_context(state: ConversationState) -> Dict[str, Any]:
    """Get current context for agents"""
    return {
        "mode": state["mode"],
        "status": state["status"],
        "has_dataset": state["dataset_id"] is not None,
        "problem_type": state["problem_type"],
        "target_variable": state["target_variable"],
        "models_trained": len(state["models_trained"]),
        "has_best_model": state["best_model"] is not None,
        "pending_approval": state["pending_approval"] is not None,
        "error_count": len(state["errors"]),
        "warning_count": len(state["warnings"]),
    }
