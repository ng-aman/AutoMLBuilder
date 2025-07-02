# src/core/events.py
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from enum import Enum
import asyncio
import json
from collections import defaultdict
from src.utils.logger import get_logger
from src.core.memory import memory

logger = get_logger(__name__)


class EventType(str, Enum):
    """Types of debug events"""

    # LLM Events
    LLM_CALL_START = "llm_call_start"
    LLM_CALL_END = "llm_call_end"
    LLM_CALL_ERROR = "llm_call_error"

    # Agent Events
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    AGENT_DECISION = "agent_decision"
    AGENT_ERROR = "agent_error"

    # Tool Events
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    TOOL_CALL_ERROR = "tool_call_error"

    # Workflow Events
    WORKFLOW_START = "workflow_start"
    WORKFLOW_END = "workflow_end"
    WORKFLOW_STATE_UPDATE = "workflow_state_update"
    WORKFLOW_ERROR = "workflow_error"

    # Human-in-the-loop Events
    APPROVAL_REQUEST = "approval_request"
    APPROVAL_RESPONSE = "approval_response"
    APPROVAL_TIMEOUT = "approval_timeout"

    # Data Processing Events
    DATA_UPLOAD = "data_upload"
    DATA_ANALYSIS = "data_analysis"
    DATA_PREPROCESSING = "data_preprocessing"
    DATA_ERROR = "data_error"

    # Model Events
    MODEL_TRAINING_START = "model_training_start"
    MODEL_TRAINING_END = "model_training_end"
    MODEL_EVALUATION = "model_evaluation"
    MODEL_ERROR = "model_error"

    # System Events
    SYSTEM_INFO = "system_info"
    SYSTEM_WARNING = "system_warning"
    SYSTEM_ERROR = "system_error"


class Event:
    """Debug event"""

    def __init__(
        self,
        event_type: EventType,
        session_id: str,
        data: Dict[str, Any],
        user_id: Optional[str] = None,
        agent: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ):
        self.event_id = self._generate_event_id()
        self.event_type = event_type
        self.session_id = session_id
        self.user_id = user_id
        self.agent = agent
        self.correlation_id = correlation_id
        self.data = data
        self.timestamp = datetime.utcnow()

    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        import uuid

        return str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "agent": self.agent,
            "correlation_id": self.correlation_id,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_json(self) -> str:
        """Convert event to JSON"""
        return json.dumps(self.to_dict())


class EventStore:
    """Store and retrieve debug events"""

    def __init__(self, max_events_per_session: int = 1000):
        self.max_events_per_session = max_events_per_session
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)

    def _get_event_key(self, session_id: str) -> str:
        """Get Redis key for events"""
        return f"events:{session_id}"

    async def add_event(self, event: Event) -> bool:
        """Add event to store"""
        try:
            key = self._get_event_key(event.session_id)

            # Get existing events
            events = await memory.backend.get(key) or []

            # Add new event
            events.append(event.to_dict())

            # Trim to max size (keep most recent)
            if len(events) > self.max_events_per_session:
                events = events[-self.max_events_per_session :]

            # Save back to Redis with 1 hour TTL
            success = await memory.backend.set(key, events, ttl=3600)

            if success:
                # Notify subscribers
                await self._notify_subscribers(event)

                # Log the event
                logger.debug(
                    f"Event: {event.event_type.value}",
                    event_id=event.event_id,
                    session_id=event.session_id,
                    agent=event.agent,
                )

            return success
        except Exception as e:
            logger.error("Failed to add event", error=str(e))
            return False

    async def get_events(
        self,
        session_id: str,
        event_types: Optional[List[EventType]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get events for a session"""
        try:
            key = self._get_event_key(session_id)
            events = await memory.backend.get(key) or []

            # Filter by event type if specified
            if event_types:
                type_values = [et.value for et in event_types]
                events = [e for e in events if e["event_type"] in type_values]

            # Apply limit
            if limit:
                events = events[-limit:]

            return events
        except Exception as e:
            logger.error("Failed to get events", session_id=session_id, error=str(e))
            return []

    async def clear_events(self, session_id: str) -> bool:
        """Clear all events for a session"""
        try:
            key = self._get_event_key(session_id)
            return await memory.backend.delete(key)
        except Exception as e:
            logger.error("Failed to clear events", session_id=session_id, error=str(e))
            return False

    def subscribe(self, session_id: str, callback: Callable):
        """Subscribe to events for a session"""
        self._subscribers[session_id].append(callback)
        logger.debug("Added event subscriber", session_id=session_id)

    def unsubscribe(self, session_id: str, callback: Callable):
        """Unsubscribe from events"""
        if callback in self._subscribers[session_id]:
            self._subscribers[session_id].remove(callback)
            logger.debug("Removed event subscriber", session_id=session_id)

    async def _notify_subscribers(self, event: Event):
        """Notify all subscribers of new event"""
        subscribers = self._subscribers.get(event.session_id, [])

        for callback in subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(
                    "Subscriber notification failed",
                    error=str(e),
                    session_id=event.session_id,
                )


class EventTracker:
    """High-level event tracking interface"""

    def __init__(self, store: Optional[EventStore] = None):
        self.store = store or EventStore()

    async def track_llm_call(
        self,
        session_id: str,
        model: str,
        prompt: str,
        response: Optional[str] = None,
        error: Optional[str] = None,
        tokens_used: Optional[int] = None,
        duration_ms: Optional[float] = None,
        user_id: Optional[str] = None,
        agent: Optional[str] = None,
    ):
        """Track LLM API call"""
        if error:
            event_type = EventType.LLM_CALL_ERROR
            data = {
                "model": model,
                "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt,
                "error": error,
            }
        else:
            event_type = EventType.LLM_CALL_END
            data = {
                "model": model,
                "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt,
                "response_preview": (
                    response[:200] + "..."
                    if response and len(response) > 200
                    else response
                ),
                "tokens_used": tokens_used,
                "duration_ms": duration_ms,
            }

        event = Event(
            event_type=event_type,
            session_id=session_id,
            data=data,
            user_id=user_id,
            agent=agent,
        )

        await self.store.add_event(event)

    async def track_agent_action(
        self,
        session_id: str,
        agent_name: str,
        action: str,
        decision: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        duration_ms: Optional[float] = None,
        user_id: Optional[str] = None,
    ):
        """Track agent action"""
        if error:
            event_type = EventType.AGENT_ERROR
            data = {"action": action, "error": error, "input_data": input_data}
        elif decision:
            event_type = EventType.AGENT_DECISION
            data = {
                "action": action,
                "decision": decision,
                "input_data": input_data,
                "output_data": output_data,
            }
        else:
            event_type = EventType.AGENT_END
            data = {
                "action": action,
                "input_data": input_data,
                "output_data": output_data,
                "duration_ms": duration_ms,
            }

        event = Event(
            event_type=event_type,
            session_id=session_id,
            data=data,
            user_id=user_id,
            agent=agent_name,
        )

        await self.store.add_event(event)

    async def track_tool_call(
        self,
        session_id: str,
        tool_name: str,
        parameters: Dict[str, Any],
        result: Optional[Any] = None,
        error: Optional[str] = None,
        duration_ms: Optional[float] = None,
        user_id: Optional[str] = None,
        agent: Optional[str] = None,
    ):
        """Track tool usage"""
        if error:
            event_type = EventType.TOOL_CALL_ERROR
            data = {"tool": tool_name, "parameters": parameters, "error": error}
        else:
            event_type = EventType.TOOL_CALL_END
            data = {
                "tool": tool_name,
                "parameters": parameters,
                "result_preview": (
                    str(result)[:200] + "..."
                    if result and len(str(result)) > 200
                    else str(result)
                ),
                "duration_ms": duration_ms,
            }

        event = Event(
            event_type=event_type,
            session_id=session_id,
            data=data,
            user_id=user_id,
            agent=agent,
        )

        await self.store.add_event(event)

    async def track_workflow_update(
        self,
        session_id: str,
        status: str,
        current_agent: Optional[str] = None,
        state_summary: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ):
        """Track workflow state update"""
        event = Event(
            event_type=EventType.WORKFLOW_STATE_UPDATE,
            session_id=session_id,
            data={
                "status": status,
                "current_agent": current_agent,
                "state_summary": state_summary,
            },
            user_id=user_id,
        )

        await self.store.add_event(event)

    async def track_model_training(
        self,
        session_id: str,
        model_name: str,
        parameters: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
        error: Optional[str] = None,
        duration_ms: Optional[float] = None,
        mlflow_run_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        """Track model training"""
        if error:
            event_type = EventType.MODEL_ERROR
            data = {"model": model_name, "parameters": parameters, "error": error}
        else:
            event_type = EventType.MODEL_TRAINING_END
            data = {
                "model": model_name,
                "parameters": parameters,
                "metrics": metrics,
                "duration_ms": duration_ms,
                "mlflow_run_id": mlflow_run_id,
            }

        event = Event(
            event_type=event_type, session_id=session_id, data=data, user_id=user_id
        )

        await self.store.add_event(event)


# Global event tracker instance
event_tracker = EventTracker()
