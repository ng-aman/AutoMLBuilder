# src/agents/__init__.py
"""
AutoML Builder Agents Module

This module contains all the intelligent agents that power the AutoML workflow.
Each agent specializes in a specific aspect of the machine learning pipeline.
"""

from src.agents.base_agent import BaseAgent
from src.agents.supervisor import SupervisorAgent
from src.agents.analysis_agent import AnalysisAgent
from src.agents.preprocessing_agent import PreprocessingAgent
from src.agents.model_agent import ModelAgent
from src.agents.optimization_agent import OptimizationAgent

# Future agents to be implemented
# from src.agents.model_agent import ModelAgent
# from src.agents.optimization_agent import OptimizationAgent

__all__ = [
    "BaseAgent",
    "SupervisorAgent",
    "AnalysisAgent",
    "PreprocessingAgent",
    "ModelAgent",
    "OptimizationAgent",
]

# Agent registry for dynamic loading
AGENT_REGISTRY = {
    "supervisor": SupervisorAgent,
    "analysis": AnalysisAgent,
    "preprocessing": PreprocessingAgent,
    # "model": ModelAgent,
    # "optimization": OptimizationAgent,
}


def get_agent(agent_name: str) -> BaseAgent:
    """
    Get an agent instance by name.

    Args:
        agent_name: Name of the agent (supervisor, analysis, etc.)

    Returns:
        Agent instance

    Raises:
        ValueError: If agent name is not found
    """
    agent_class = AGENT_REGISTRY.get(agent_name.lower())
    if not agent_class:
        raise ValueError(f"Unknown agent: {agent_name}")

    return agent_class()
