# src/frontend/pages/debug.py
import streamlit as st
import pandas as pd
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import asyncio
import websocket
import threading


def render(session_state: Dict[str, Any]):
    """Render the debug page"""
    st.title("🐛 Debug Console")
    st.markdown("Real-time view of agent activities and system events")

    api_client = session_state.get("api_client")
    session_manager = session_state.get("session_manager")

    if not api_client or not session_manager:
        st.error("Session not initialized properly")
        return

    # Check if there's an active session
    current_session_id = st.session_state.get("current_session_id")
    if not current_session_id:
        st.warning("No active session. Start a chat to see debug information.")
        return

    # Debug controls
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        auto_refresh = st.checkbox("Auto-refresh", value=True)

    with col2:
        refresh_rate = st.select_slider(
            "Refresh rate",
            options=[1, 2, 5, 10],
            value=2,
            format_func=lambda x: f"{x}s",
        )

    with col3:
        if st.button("🔄 Refresh Now"):
            st.rerun()

    with col4:
        if st.button("🗑️ Clear Events"):
            # This would clear events via API
            st.info("Events cleared")

    # Event filters
    st.subheader("🔍 Event Filters")

    col1, col2 = st.columns(2)

    with col1:
        event_types = st.multiselect(
            "Event Types",
            [
                "llm_call_start",
                "llm_call_end",
                "llm_call_error",
                "agent_start",
                "agent_end",
                "agent_decision",
                "agent_error",
                "tool_call_start",
                "tool_call_end",
                "tool_call_error",
                "workflow_start",
                "workflow_end",
                "workflow_state_update",
                "approval_request",
                "approval_response",
                "model_training_start",
                "model_training_end",
            ],
            default=["agent_decision", "llm_call_end", "tool_call_end"],
        )

    with col2:
        limit = st.number_input("Max events", min_value=10, max_value=500, value=100)

    # Get debug events
    events_data = api_client.get_debug_events(
        session_id=current_session_id,
        event_types=event_types if event_types else None,
        limit=limit,
    )

    if events_data and events_data.get("events"):
        render_events_timeline(events_data["events"])
        render_events_details(events_data["events"])
        render_events_statistics(events_data["events"])
    else:
        st.info(
            "No events found. Make sure debug mode is enabled and interact with the chat."
        )

    # Auto-refresh
    if auto_refresh:
        st.empty()
        import time

        time.sleep(refresh_rate)
        st.rerun()


def render_events_timeline(events: List[Dict[str, Any]]):
    """Render events timeline"""
    st.subheader("📊 Events Timeline")

    # Create timeline visualization
    timeline_data = []

    for event in reversed(events[-20:]):  # Show last 20 events
        event_type = event["event_type"]
        timestamp = event["timestamp"]

        # Determine color based on event type
        if "error" in event_type:
            color = "🔴"
        elif "start" in event_type:
            color = "🟡"
        elif "end" in event_type or "decision" in event_type:
            color = "🟢"
        else:
            color = "🔵"

        # Format event for display
        if event_type == "agent_decision":
            description = f"{event['data'].get('decision', 'Unknown decision')}"
        elif event_type == "llm_call_end":
            description = f"LLM: {event['data'].get('model', 'unknown')}"
        elif event_type == "tool_call_end":
            description = f"Tool: {event['data'].get('tool', 'unknown')}"
        else:
            description = event_type.replace("_", " ").title()

        timeline_data.append(
            {
                "Time": timestamp[11:19],  # Extract time portion
                "Event": f"{color} {description}",
                "Agent": event.get("agent", "System"),
                "Duration": (
                    f"{event['data'].get('duration_ms', 0):.0f}ms"
                    if "duration_ms" in event.get("data", {})
                    else "-"
                ),
            }
        )

    # Display as dataframe
    if timeline_data:
        df = pd.DataFrame(timeline_data)
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_events_details(events: List[Dict[str, Any]]):
    """Render detailed event information"""
    st.subheader("🔍 Event Details")

    # Group events by type
    event_groups = {}
    for event in events:
        event_type = event["event_type"]
        if event_type not in event_groups:
            event_groups[event_type] = []
        event_groups[event_type].append(event)

    # Create tabs for each event type
    if event_groups:
        tabs = st.tabs(list(event_groups.keys()))

        for tab, event_type in zip(tabs, event_groups.keys()):
            with tab:
                render_event_group(event_groups[event_type])


def render_event_group(events: List[Dict[str, Any]]):
    """Render a group of events of the same type"""
    for i, event in enumerate(reversed(events[-10:])):  # Show last 10 of each type
        with st.expander(
            f"{event['timestamp'][11:19]} - {event.get('agent', 'System')}",
            expanded=i == 0,  # Expand first item
        ):
            # Event metadata
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Event ID**: {event['event_id'][:8]}...")
                st.write(f"**Session**: {event['session_id'][:8]}...")
                if event.get("correlation_id"):
                    st.write(f"**Correlation**: {event['correlation_id'][:8]}...")

            with col2:
                st.write(f"**Timestamp**: {event['timestamp']}")
                if event.get("user_id"):
                    st.write(f"**User**: {event['user_id'][:8]}...")
                if event.get("agent"):
                    st.write(f"**Agent**: {event['agent']}")

            # Event data
            st.write("**Event Data**:")

            # Special formatting for different event types
            if event["event_type"] == "llm_call_end":
                render_llm_event(event["data"])
            elif event["event_type"] == "agent_decision":
                render_agent_decision(event["data"])
            elif event["event_type"] == "tool_call_end":
                render_tool_call(event["data"])
            else:
                # Generic JSON display
                st.json(event["data"])


def render_llm_event(data: Dict[str, Any]):
    """Render LLM call event"""
    if data.get("prompt_preview"):
        st.text_area("Prompt", data["prompt_preview"], height=100, disabled=True)

    if data.get("response_preview"):
        st.text_area("Response", data["response_preview"], height=100, disabled=True)

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", data.get("model", "Unknown"))
    with col2:
        st.metric("Tokens", data.get("tokens_used", "N/A"))
    with col3:
        st.metric("Duration", f"{data.get('duration_ms', 0):.0f}ms")


def render_agent_decision(data: Dict[str, Any]):
    """Render agent decision event"""
    st.write(f"**Action**: {data.get('action', 'Unknown')}")
    st.write(f"**Decision**: {data.get('decision', 'Unknown')}")

    if data.get("input_data"):
        with st.expander("Input Data"):
            st.json(data["input_data"])

    if data.get("output_data"):
        with st.expander("Output Data"):
            st.json(data["output_data"])


def render_tool_call(data: Dict[str, Any]):
    """Render tool call event"""
    st.write(f"**Tool**: {data.get('tool', 'Unknown')}")

    if data.get("parameters"):
        with st.expander("Parameters"):
            st.json(data["parameters"])

    if data.get("result_preview"):
        st.text_area("Result", data["result_preview"], height=100, disabled=True)

    if data.get("duration_ms"):
        st.metric("Duration", f"{data['duration_ms']:.0f}ms")


def render_events_statistics(events: List[Dict[str, Any]]):
    """Render event statistics"""
    st.subheader("📈 Event Statistics")

    col1, col2 = st.columns(2)

    with col1:
        # Event type distribution
        st.write("**Event Type Distribution**")

        type_counts = {}
        for event in events:
            event_type = event["event_type"]
            type_counts[event_type] = type_counts.get(event_type, 0) + 1

        if type_counts:
            df = pd.DataFrame(
                list(type_counts.items()), columns=["Event Type", "Count"]
            ).sort_values("Count", ascending=False)

            st.bar_chart(df.set_index("Event Type"))

    with col2:
        # Agent activity
        st.write("**Agent Activity**")

        agent_counts = {}
        for event in events:
            if event.get("agent"):
                agent = event["agent"]
                agent_counts[agent] = agent_counts.get(agent, 0) + 1

        if agent_counts:
            df = pd.DataFrame(
                list(agent_counts.items()), columns=["Agent", "Events"]
            ).sort_values("Events", ascending=False)

            st.bar_chart(df.set_index("Agent"))

    # Performance metrics
    st.write("**Performance Metrics**")

    col1, col2, col3, col4 = st.columns(4)

    # Calculate metrics
    llm_calls = [e for e in events if e["event_type"] == "llm_call_end"]
    tool_calls = [e for e in events if e["event_type"] == "tool_call_end"]
    errors = [e for e in events if "error" in e["event_type"]]

    with col1:
        st.metric("Total Events", len(events))

    with col2:
        st.metric("LLM Calls", len(llm_calls))

    with col3:
        st.metric("Tool Calls", len(tool_calls))

    with col4:
        st.metric("Errors", len(errors))

    # Average durations
    if llm_calls:
        avg_llm_duration = sum(
            e["data"].get("duration_ms", 0) for e in llm_calls
        ) / len(llm_calls)
        st.write(f"**Avg LLM Duration**: {avg_llm_duration:.0f}ms")

    if tool_calls:
        avg_tool_duration = sum(
            e["data"].get("duration_ms", 0) for e in tool_calls
        ) / len(tool_calls)
        st.write(f"**Avg Tool Duration**: {avg_tool_duration:.0f}ms")


# WebSocket connection for real-time events (optional enhancement)
class EventStreamClient:
    """WebSocket client for real-time event streaming"""

    def __init__(self, websocket_url: str, on_event_callback):
        self.url = websocket_url
        self.on_event = on_event_callback
        self.ws = None
        self.running = False

    def start(self):
        """Start WebSocket connection"""
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop WebSocket connection"""
        self.running = False
        if self.ws:
            self.ws.close()

    def _run(self):
        """Run WebSocket client"""

        def on_message(ws, message):
            try:
                event = json.loads(message)
                self.on_event(event)
            except:
                pass

        def on_error(ws, error):
            st.error(f"WebSocket error: {error}")

        self.ws = websocket.WebSocketApp(
            self.url, on_message=on_message, on_error=on_error
        )

        while self.running:
            try:
                self.ws.run_forever()
            except:
                pass
