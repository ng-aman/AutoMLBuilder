"""
Debug console component for AutoML Builder.

This module provides a real-time debug console for monitoring the multi-agent
system's execution, events, and internal state.
"""

import streamlit as st
from typing import Dict, List, Optional, Any, Callable
import json
import asyncio
import websockets
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field
import threading
import time

from ..utils.api_client import APIClient
from ...core.events import EventType


@dataclass
class DebugEvent:
    """Debug event model."""

    timestamp: datetime
    event_type: str
    source: str  # Agent or component name
    level: str  # "info", "warning", "error", "debug"
    message: str
    data: Optional[Dict[str, Any]] = None
    correlation_id: Optional[str] = None
    duration_ms: Optional[int] = None


class DebugConsole:
    """
    Streamlit debug console component.

    Provides real-time monitoring of agent execution, events, and system state.
    """

    def __init__(
        self,
        api_client: APIClient,
        session_state: Any,
        websocket_url: str = "ws://localhost:8000/api/debug/stream",
        max_events: int = 1000,
    ):
        """
        Initialize debug console.

        Args:
            api_client: API client instance
            session_state: Streamlit session state
            websocket_url: WebSocket URL for event streaming
            max_events: Maximum number of events to store
        """
        self.api_client = api_client
        self.session_state = session_state
        self.websocket_url = websocket_url
        self.max_events = max_events

        # Initialize session state
        self._initialize_session_state()

        # WebSocket connection
        self.ws_connection = None
        self.ws_thread = None

    def _initialize_session_state(self):
        """Initialize session state variables."""
        if "debug_events" not in self.session_state:
            self.session_state.debug_events = deque(maxlen=self.max_events)

        if "debug_filters" not in self.session_state:
            self.session_state.debug_filters = {
                "levels": ["info", "warning", "error"],
                "sources": [],
                "search": "",
            }

        if "debug_paused" not in self.session_state:
            self.session_state.debug_paused = False

        if "debug_connected" not in self.session_state:
            self.session_state.debug_connected = False

        if "event_stats" not in self.session_state:
            self.session_state.event_stats = {
                "total": 0,
                "by_level": {},
                "by_source": {},
                "by_type": {},
            }

    def render(self):
        """Render the debug console."""
        # Console header
        self._render_header()

        # Filters and controls
        self._render_controls()

        # Event statistics
        self._render_statistics()

        # Event stream
        self._render_event_stream()

        # Event details
        self._render_event_details()

    def _render_header(self):
        """Render console header."""
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown("### 🐛 Debug Console")

        with col2:
            status = (
                "🟢 Connected"
                if self.session_state.debug_connected
                else "🔴 Disconnected"
            )
            st.markdown(f"**Status:** {status}")

        with col3:
            if self.session_state.debug_connected:
                if st.button("Disconnect", type="secondary"):
                    self._disconnect_websocket()
            else:
                if st.button("Connect", type="primary"):
                    self._connect_websocket()

    def _render_controls(self):
        """Render filter controls."""
        with st.expander("🔧 Filters & Controls", expanded=True):
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

            with col1:
                # Level filter
                levels = st.multiselect(
                    "Log Levels",
                    ["debug", "info", "warning", "error"],
                    default=self.session_state.debug_filters["levels"],
                    key="debug_level_filter",
                )
                self.session_state.debug_filters["levels"] = levels

            with col2:
                # Source filter
                available_sources = list(
                    self.session_state.event_stats["by_source"].keys()
                )
                sources = st.multiselect(
                    "Sources",
                    available_sources,
                    default=self.session_state.debug_filters["sources"],
                    key="debug_source_filter",
                    help="Filter by agent or component",
                )
                self.session_state.debug_filters["sources"] = sources

            with col3:
                # Search filter
                search = st.text_input(
                    "Search",
                    value=self.session_state.debug_filters["search"],
                    key="debug_search_filter",
                    placeholder="Filter messages...",
                )
                self.session_state.debug_filters["search"] = search

            with col4:
                # Controls
                st.markdown("&nbsp;")  # Spacing
                col_a, col_b = st.columns(2)

                with col_a:
                    if st.button("⏸️" if not self.session_state.debug_paused else "▶️"):
                        self.session_state.debug_paused = (
                            not self.session_state.debug_paused
                        )

                with col_b:
                    if st.button("🗑️", help="Clear events"):
                        self._clear_events()

    def _render_statistics(self):
        """Render event statistics."""
        stats = self.session_state.event_stats

        with st.container():
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Events", stats["total"])

            with col2:
                errors = stats["by_level"].get("error", 0)
                warnings = stats["by_level"].get("warning", 0)
                st.metric(
                    "Errors/Warnings",
                    f"{errors}/{warnings}",
                    delta=errors if errors > 0 else None,
                )

            with col3:
                active_agents = len(stats["by_source"])
                st.metric("Active Agents", active_agents)

            with col4:
                if stats["total"] > 0:
                    avg_duration = (
                        sum(
                            e.duration_ms
                            for e in self.session_state.debug_events
                            if e.duration_ms
                        )
                        / stats["total"]
                    )
                    st.metric("Avg Duration", f"{avg_duration:.0f}ms")
                else:
                    st.metric("Avg Duration", "N/A")

    def _render_event_stream(self):
        """Render the event stream."""
        st.markdown("#### Event Stream")

        # Create container for events
        event_container = st.container()

        with event_container:
            # Filter events
            filtered_events = self._filter_events()

            # Display events in reverse order (newest first)
            for event in reversed(filtered_events):
                self._render_event(event)

            # Show message if no events
            if not filtered_events:
                st.info("No events matching current filters")

    def _render_event(self, event: DebugEvent):
        """
        Render a single event.

        Args:
            event: Debug event to render
        """
        # Determine styling based on level
        level_colors = {
            "debug": "#6c757d",
            "info": "#0dcaf0",
            "warning": "#ffc107",
            "error": "#dc3545",
        }

        level_icons = {"debug": "🔍", "info": "ℹ️", "warning": "⚠️", "error": "❌"}

        color = level_colors.get(event.level, "#6c757d")
        icon = level_icons.get(event.level, "📝")

        # Create event display
        with st.container():
            col1, col2, col3, col4 = st.columns([1, 2, 6, 1])

            with col1:
                st.markdown(
                    f'<span style="color: {color};">{icon} **{event.level.upper()}**</span>',
                    unsafe_allow_html=True,
                )

            with col2:
                st.text(f"{event.timestamp.strftime('%H:%M:%S.%f')[:-3]}")
                st.caption(event.source)

            with col3:
                st.text(event.message)

                # Show event type if different from level
                if event.event_type and event.event_type != event.level:
                    st.caption(f"Type: {event.event_type}")

            with col4:
                if event.duration_ms:
                    st.caption(f"{event.duration_ms}ms")

                if event.data:
                    if st.button("📋", key=f"event_{id(event)}", help="View details"):
                        self.session_state.selected_event = event

            # Separator
            st.markdown("---")

    def _render_event_details(self):
        """Render detailed view of selected event."""
        if (
            hasattr(self.session_state, "selected_event")
            and self.session_state.selected_event
        ):
            event = self.session_state.selected_event

            with st.expander("📋 Event Details", expanded=True):
                # Event metadata
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Event Information**")
                    st.text(f"Timestamp: {event.timestamp}")
                    st.text(f"Type: {event.event_type}")
                    st.text(f"Source: {event.source}")
                    st.text(f"Level: {event.level}")
                    if event.correlation_id:
                        st.text(f"Correlation ID: {event.correlation_id}")
                    if event.duration_ms:
                        st.text(f"Duration: {event.duration_ms}ms")

                with col2:
                    st.markdown("**Message**")
                    st.code(event.message, language="text")

                # Event data
                if event.data:
                    st.markdown("**Event Data**")
                    st.json(event.data)

                # Close button
                if st.button("Close", type="secondary"):
                    self.session_state.selected_event = None

    def _filter_events(self) -> List[DebugEvent]:
        """
        Filter events based on current filters.

        Returns:
            Filtered list of events
        """
        filters = self.session_state.debug_filters
        events = list(self.session_state.debug_events)

        # Level filter
        if filters["levels"]:
            events = [e for e in events if e.level in filters["levels"]]

        # Source filter
        if filters["sources"]:
            events = [e for e in events if e.source in filters["sources"]]

        # Search filter
        if filters["search"]:
            search_lower = filters["search"].lower()
            events = [
                e
                for e in events
                if search_lower in e.message.lower()
                or search_lower in e.source.lower()
                or (e.event_type and search_lower in e.event_type.lower())
            ]

        return events

    def _connect_websocket(self):
        """Connect to WebSocket for event streaming."""
        if self.ws_thread and self.ws_thread.is_alive():
            return

        self.session_state.debug_connected = True
        self.ws_thread = threading.Thread(target=self._websocket_listener, daemon=True)
        self.ws_thread.start()

    def _disconnect_websocket(self):
        """Disconnect WebSocket."""
        self.session_state.debug_connected = False
        if self.ws_connection:
            asyncio.run(self.ws_connection.close())

    def _websocket_listener(self):
        """WebSocket listener thread."""

        async def listen():
            try:
                # Add auth token to URL if available
                url = self.websocket_url
                if hasattr(self.session_state, "auth_token"):
                    url += f"?token={self.session_state.auth_token}"

                async with websockets.connect(url) as websocket:
                    self.ws_connection = websocket

                    while self.session_state.debug_connected:
                        try:
                            message = await asyncio.wait_for(
                                websocket.recv(), timeout=1.0
                            )

                            # Parse and add event
                            event_data = json.loads(message)
                            event = DebugEvent(
                                timestamp=datetime.fromisoformat(
                                    event_data["timestamp"]
                                ),
                                event_type=event_data["event_type"],
                                source=event_data["source"],
                                level=event_data["level"],
                                message=event_data["message"],
                                data=event_data.get("data"),
                                correlation_id=event_data.get("correlation_id"),
                                duration_ms=event_data.get("duration_ms"),
                            )

                            # Add event if not paused
                            if not self.session_state.debug_paused:
                                self.session_state.debug_events.append(event)
                                self._update_statistics(event)

                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            print(f"WebSocket error: {e}")
                            break

            except Exception as e:
                print(f"WebSocket connection error: {e}")
                self.session_state.debug_connected = False

        # Run async listener
        asyncio.run(listen())

    def _update_statistics(self, event: DebugEvent):
        """
        Update event statistics.

        Args:
            event: New event to include in statistics
        """
        stats = self.session_state.event_stats

        # Total count
        stats["total"] += 1

        # By level
        if event.level not in stats["by_level"]:
            stats["by_level"][event.level] = 0
        stats["by_level"][event.level] += 1

        # By source
        if event.source not in stats["by_source"]:
            stats["by_source"][event.source] = 0
        stats["by_source"][event.source] += 1

        # By type
        if event.event_type not in stats["by_type"]:
            stats["by_type"][event.event_type] = 0
        stats["by_type"][event.event_type] += 1

    def _clear_events(self):
        """Clear all events and reset statistics."""
        self.session_state.debug_events.clear()
        self.session_state.event_stats = {
            "total": 0,
            "by_level": {},
            "by_source": {},
            "by_type": {},
        }
        self.session_state.selected_event = None

    def add_event(self, event: DebugEvent):
        """
        Manually add an event to the console.

        Args:
            event: Event to add
        """
        if not self.session_state.debug_paused:
            self.session_state.debug_events.append(event)
            self._update_statistics(event)


# Helper function for creating debug console
def create_debug_console(
    api_client: APIClient, session_state: Any, websocket_url: Optional[str] = None
) -> DebugConsole:
    """
    Create and return a debug console instance.

    Args:
        api_client: API client instance
        session_state: Streamlit session state
        websocket_url: Optional WebSocket URL override

    Returns:
        DebugConsole instance
    """
    return DebugConsole(
        api_client,
        session_state,
        websocket_url=websocket_url or "ws://localhost:8000/api/debug/stream",
    )
