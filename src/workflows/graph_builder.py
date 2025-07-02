# src/workflows/graph_builder.py
from typing import Dict, Any, List, Callable
from langgraph.graph import Graph, StateGraph, END
from langgraph.checkpoint import MemorySaver
from langgraph.prebuilt import ToolNode
from src.core.state import (
    ConversationState,
    WorkflowStatus,
    AgentType,
    is_workflow_complete,
)
from src.agents.supervisor import SupervisorAgent
from src.agents.analysis_agent import AnalysisAgent
from src.agents.preprocessing_agent import PreprocessingAgent
from src.workflows.nodes import (
    supervisor_node,
    analysis_node,
    preprocessing_node,
    model_node,
    optimization_node,
    human_approval_node,
    error_handler_node,
)
from src.workflows.edges import (
    should_continue,
    route_supervisor,
    route_after_analysis,
    route_after_preprocessing,
    route_after_model,
    route_after_optimization,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_workflow_graph() -> StateGraph:
    """
    Create the main workflow graph for the AutoML system.
    This graph orchestrates all agents and manages the workflow.
    """

    # Create the graph with ConversationState
    workflow = StateGraph(ConversationState)

    # Add nodes for each agent
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("preprocessing", preprocessing_node)
    workflow.add_node("model", model_node)
    workflow.add_node("optimization", optimization_node)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("error_handler", error_handler_node)

    # Set entry point
    workflow.set_entry_point("supervisor")

    # Add edges with conditional routing

    # From supervisor
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "analysis": "analysis",
            "preprocessing": "preprocessing",
            "model": "model",
            "optimization": "optimization",
            "human_approval": "human_approval",
            "error": "error_handler",
            "end": END,
        },
    )

    # From analysis
    workflow.add_conditional_edges(
        "analysis",
        route_after_analysis,
        {
            "supervisor": "supervisor",
            "preprocessing": "preprocessing",
            "human_approval": "human_approval",
            "error": "error_handler",
        },
    )

    # From preprocessing
    workflow.add_conditional_edges(
        "preprocessing",
        route_after_preprocessing,
        {
            "supervisor": "supervisor",
            "model": "model",
            "human_approval": "human_approval",
            "error": "error_handler",
        },
    )

    # From model
    workflow.add_conditional_edges(
        "model",
        route_after_model,
        {
            "supervisor": "supervisor",
            "optimization": "optimization",
            "human_approval": "human_approval",
            "error": "error_handler",
            "end": END,
        },
    )

    # From optimization
    workflow.add_conditional_edges(
        "optimization",
        route_after_optimization,
        {
            "supervisor": "supervisor",
            "human_approval": "human_approval",
            "error": "error_handler",
            "end": END,
        },
    )

    # From human approval
    workflow.add_conditional_edges(
        "human_approval",
        lambda x: "supervisor",  # Always go back to supervisor after approval
        {"supervisor": "supervisor"},
    )

    # From error handler
    workflow.add_conditional_edges(
        "error_handler",
        lambda x: "end" if x["errors"] and len(x["errors"]) > 3 else "supervisor",
        {"supervisor": "supervisor", "end": END},
    )

    # Compile the graph
    compiled_graph = workflow.compile()

    logger.info("Workflow graph created successfully")

    return compiled_graph


def create_simple_workflow() -> StateGraph:
    """
    Create a simplified workflow for testing or specific use cases.
    """
    workflow = StateGraph(ConversationState)

    # Add only essential nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("model", model_node)

    # Simple flow
    workflow.set_entry_point("supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        lambda x: (
            "analysis"
            if x["dataset_id"] and not x["dataset_info"]
            else "model" if x["dataset_info"] else "end"
        ),
        {"analysis": "analysis", "model": "model", "end": END},
    )

    workflow.add_edge("analysis", "supervisor")
    workflow.add_edge("model", END)

    return workflow.compile()


def create_interactive_workflow() -> StateGraph:
    """
    Create an interactive workflow that requires human approval at each step.
    """
    workflow = StateGraph(ConversationState)

    # Add all nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("preprocessing", preprocessing_node)
    workflow.add_node("model", model_node)
    workflow.add_node("optimization", optimization_node)
    workflow.add_node("human_approval", human_approval_node)

    # Set entry point
    workflow.set_entry_point("supervisor")

    # Add edges that always go through human approval
    def route_with_approval(state: ConversationState) -> str:
        """Route to human approval if in interactive mode"""
        if state["mode"] == "interactive" and state.get("pending_approval"):
            return "human_approval"
        return route_supervisor(state)

    workflow.add_conditional_edges(
        "supervisor",
        route_with_approval,
        {
            "analysis": "analysis",
            "preprocessing": "preprocessing",
            "model": "model",
            "optimization": "optimization",
            "human_approval": "human_approval",
            "end": END,
        },
    )

    # All agents route back through supervisor for approval checks
    for node in ["analysis", "preprocessing", "model", "optimization"]:
        workflow.add_edge(node, "supervisor")

    workflow.add_edge("human_approval", "supervisor")

    return workflow.compile()


def visualize_workflow(graph: StateGraph) -> str:
    """
    Generate a visual representation of the workflow graph.
    Returns a Mermaid diagram string.
    """
    mermaid = """
    graph TD
        Start[Start] --> Supervisor{Supervisor}
        Supervisor --> |Dataset Analysis| Analysis[Analysis Agent]
        Supervisor --> |Data Preprocessing| Preprocessing[Preprocessing Agent]
        Supervisor --> |Model Training| Model[Model Agent]
        Supervisor --> |Hyperparameter Tuning| Optimization[Optimization Agent]
        Supervisor --> |Needs Approval| Approval[Human Approval]
        Supervisor --> |Error| Error[Error Handler]
        Supervisor --> |Complete| End[End]
        
        Analysis --> Supervisor
        Preprocessing --> Supervisor
        Model --> Supervisor
        Optimization --> Supervisor
        Approval --> Supervisor
        Error --> Supervisor
        Error --> |Too Many Errors| End
        
        style Supervisor fill:#f9f,stroke:#333,stroke-width:4px
        style Analysis fill:#bbf,stroke:#333,stroke-width:2px
        style Preprocessing fill:#bbf,stroke:#333,stroke-width:2px
        style Model fill:#bbf,stroke:#333,stroke-width:2px
        style Optimization fill:#bbf,stroke:#333,stroke-width:2px
        style Approval fill:#fbf,stroke:#333,stroke-width:2px
        style Error fill:#fbb,stroke:#333,stroke-width:2px
    """
    return mermaid


# Export the main graph creation function
__all__ = [
    "create_workflow_graph",
    "create_simple_workflow",
    "create_interactive_workflow",
    "visualize_workflow",
]
