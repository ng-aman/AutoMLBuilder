# src/__init__.py
"""
AutoML Builder

An intelligent, production-ready AutoML platform powered by LangGraph multi-agent architecture.
Build, train, and optimize machine learning models through natural language conversations.
"""

__version__ = "1.0.0"
__author__ = "AutoML Builder Team"
__license__ = "MIT"

# Core components
from src.core import settings

# Agents
from src.agents import SupervisorAgent, AnalysisAgent, PreprocessingAgent

# API
from src.api import app

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    # Core
    "settings",
    # Agents
    "SupervisorAgent",
    "AnalysisAgent",
    "PreprocessingAgent",
    # API
    "app",
]
