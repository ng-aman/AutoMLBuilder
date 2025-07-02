# src/core/__init__.py
"""
AutoML Builder Core Module

Core functionality including configuration, state management, memory, and event tracking.
"""

from src.core.config import settings
from src.core.state import (
    ConversationState,
    WorkflowMode,
    WorkflowStatus,
    ProblemType,
    AgentType,
    create_initial_state,
)
from src.core.memory import memory, ConversationMemory
from src.core.events import event_tracker, EventType, Event

__all__ = [
    # Configuration
    "settings",
    # State management
    "ConversationState",
    "WorkflowMode",
    "WorkflowStatus",
    "ProblemType",
    "AgentType",
    "create_initial_state",
    # Memory
    "memory",
    "ConversationMemory",
    # Events
    "event_tracker",
    "EventType",
    "Event",
]
