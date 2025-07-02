# src/workflows/__init__.py
"""
AutoML Builder Workflows Module

LangGraph-based workflow orchestration for the multi-agent system.
Defines the graph structure, nodes, and edges for the AutoML pipeline.
"""

from src.workflows.graph_builder import (
    create_workflow_graph,
    create_simple_workflow,
    create_interactive_workflow,
    visualize_workflow,
)

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

__all__ = [
    # Graph builders
    "create_workflow_graph",
    "create_simple_workflow",
    "create_interactive_workflow",
    "visualize_workflow",
    # Nodes
    "supervisor_node",
    "analysis_node",
    "preprocessing_node",
    "model_node",
    "optimization_node",
    "human_approval_node",
    "error_handler_node",
    # Edges
    "should_continue",
    "route_supervisor",
    "route_after_analysis",
    "route_after_preprocessing",
    "route_after_model",
    "route_after_optimization",
]
