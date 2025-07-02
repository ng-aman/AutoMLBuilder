"""
Chat interface component for AutoML Builder.

This module provides the interactive chat interface for users to interact
with the multi-agent system.
"""

import streamlit as st
from typing import Dict, List, Optional, Any, Callable
import json
import time
from datetime import datetime
import asyncio
from dataclasses import dataclass, asdict

from ..utils.api_client import APIClient
from ...core.events import EventType


@dataclass
class Message:
    """Chat message model."""

    role: str  # "user", "assistant", "system", "agent"
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    agent_name: Optional[str] = None
    message_type: Optional[str] = None  # "text", "code", "data", "visualization"


class ChatInterface:
    """
    Streamlit chat interface component.

    Provides an interactive chat interface for communicating with the
    AutoML multi-agent system.
    """

    def __init__(
        self,
        api_client: APIClient,
        session_state: Any,
        on_message_callback: Optional[Callable] = None,
    ):
        """
        Initialize chat interface.

        Args:
            api_client: API client instance
            session_state: Streamlit session state
            on_message_callback: Optional callback for message events
        """
        self.api_client = api_client
        self.session_state = session_state
        self.on_message_callback = on_message_callback

        # Initialize session state
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize session state variables."""
        if "messages" not in self.session_state:
            self.session_state.messages = []

        if "conversation_id" not in self.session_state:
            self.session_state.conversation_id = None

        if "is_processing" not in self.session_state:
            self.session_state.is_processing = False

        if "pending_approval" not in self.session_state:
            self.session_state.pending_approval = None

        if "chat_mode" not in self.session_state:
            self.session_state.chat_mode = "interactive"  # "interactive" or "auto"

    def render(self):
        """Render the chat interface."""
        # Chat header
        self._render_header()

        # Chat messages
        self._render_messages()

        # Pending approval section
        if self.session_state.pending_approval:
            self._render_approval_section()

        # Chat input
        self._render_input()

        # Chat controls
        self._render_controls()

    def _render_header(self):
        """Render chat header with mode selector."""
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown("### 💬 AutoML Assistant")

        with col2:
            mode = st.selectbox(
                "Mode",
                ["interactive", "auto"],
                index=0 if self.session_state.chat_mode == "interactive" else 1,
                key="chat_mode_selector",
                help="Interactive: Approve each step | Auto: Run automatically",
            )
            if mode != self.session_state.chat_mode:
                self.session_state.chat_mode = mode
                self._add_system_message(f"Switched to {mode} mode")

        with col3:
            if st.button("Clear Chat", type="secondary"):
                self._clear_chat()

    def _render_messages(self):
        """Render chat messages."""
        # Create a container for messages
        message_container = st.container()

        with message_container:
            for message in self.session_state.messages:
                self._render_message(message)

        # Auto-scroll to bottom
        if self.session_state.is_processing:
            st.empty()  # Force update

    def _render_message(self, message: Message):
        """
        Render a single message.

        Args:
            message: Message to render
        """
        # Determine avatar and name
        if message.role == "user":
            avatar = "👤"
            name = "You"
        elif message.role == "assistant":
            avatar = "🤖"
            name = "Assistant"
        elif message.role == "agent":
            avatar = self._get_agent_avatar(message.agent_name)
            name = message.agent_name or "Agent"
        else:
            avatar = "ℹ️"
            name = "System"

        # Create message layout
        with st.chat_message(message.role, avatar=avatar):
            # Message header
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{name}**")
            with col2:
                st.caption(message.timestamp.strftime("%H:%M:%S"))

            # Message content based on type
            if message.message_type == "code":
                st.code(message.content, language="python")
            elif message.message_type == "data":
                self._render_data_message(message.content)
            elif message.message_type == "visualization":
                self._render_visualization_message(message.content)
            else:
                st.markdown(message.content)

            # Metadata
            if message.metadata:
                with st.expander("Details"):
                    st.json(message.metadata)

    def _render_data_message(self, content: str):
        """Render data message with table preview."""
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "preview" in data:
                st.dataframe(data["preview"], use_container_width=True)
                if "info" in data:
                    st.info(data["info"])
            else:
                st.json(data)
        except:
            st.text(content)

    def _render_visualization_message(self, content: str):
        """Render visualization message."""
        try:
            viz_data = json.loads(content)
            if "type" in viz_data:
                if viz_data["type"] == "plotly":
                    st.plotly_chart(viz_data["figure"], use_container_width=True)
                elif viz_data["type"] == "metric":
                    cols = st.columns(len(viz_data["metrics"]))
                    for i, metric in enumerate(viz_data["metrics"]):
                        with cols[i]:
                            st.metric(
                                label=metric["label"],
                                value=metric["value"],
                                delta=metric.get("delta"),
                            )
                else:
                    st.json(viz_data)
        except:
            st.text(content)

    def _render_approval_section(self):
        """Render pending approval section."""
        approval = self.session_state.pending_approval

        with st.container():
            st.warning("🔔 **Action Required**")
            st.markdown(f"**{approval['agent']}** wants to: {approval['action']}")

            if "details" in approval:
                with st.expander("View Details"):
                    st.json(approval["details"])

            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                if st.button("✅ Approve", type="primary"):
                    self._handle_approval(True)

            with col2:
                if st.button("❌ Reject", type="secondary"):
                    self._handle_approval(False)

            with col3:
                feedback = st.text_input(
                    "Feedback (optional)",
                    key="approval_feedback",
                    placeholder="Provide guidance...",
                )

    def _render_input(self):
        """Render chat input."""
        # Disable input while processing
        disabled = self.session_state.is_processing

        # Create input form
        with st.form("chat_input_form", clear_on_submit=True):
            col1, col2 = st.columns([5, 1])

            with col1:
                user_input = st.text_area(
                    "Message",
                    key="chat_input",
                    placeholder="Ask me anything about your ML project...",
                    disabled=disabled,
                    height=100,
                )

            with col2:
                submitted = st.form_submit_button(
                    "Send", type="primary", disabled=disabled, use_container_width=True
                )

            # Quick actions
            st.markdown("**Quick Actions:**")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if st.form_submit_button("📊 Analyze Data"):
                    user_input = "Analyze my dataset and show me insights"
                    submitted = True

            with col2:
                if st.form_submit_button("🧹 Preprocess"):
                    user_input = "Clean and preprocess my data"
                    submitted = True

            with col3:
                if st.form_submit_button("🤖 Train Models"):
                    user_input = "Train and compare different models"
                    submitted = True

            with col4:
                if st.form_submit_button("⚡ Optimize"):
                    user_input = "Optimize hyperparameters for the best model"
                    submitted = True

        # Handle submission
        if submitted and user_input:
            self._handle_user_message(user_input)

    def _render_controls(self):
        """Render chat controls."""
        with st.expander("⚙️ Chat Settings"):
            col1, col2 = st.columns(2)

            with col1:
                st.checkbox(
                    "Show agent thinking process",
                    key="show_agent_thinking",
                    help="Display internal agent reasoning",
                )

                st.checkbox(
                    "Enable notifications",
                    key="enable_notifications",
                    help="Show desktop notifications for important events",
                )

            with col2:
                st.number_input(
                    "Response timeout (seconds)",
                    min_value=30,
                    max_value=600,
                    value=120,
                    key="response_timeout",
                )

                st.selectbox(
                    "Response detail level",
                    ["concise", "normal", "detailed"],
                    index=1,
                    key="response_detail",
                )

    def _handle_user_message(self, content: str):
        """
        Handle user message submission.

        Args:
            content: User message content
        """
        # Add user message
        user_message = Message(role="user", content=content, timestamp=datetime.now())
        self.session_state.messages.append(user_message)

        # Set processing state
        self.session_state.is_processing = True

        # Send message to backend
        asyncio.run(self._send_message_async(content))

    async def _send_message_async(self, content: str):
        """
        Send message to backend asynchronously.

        Args:
            content: Message content
        """
        try:
            # Create or get conversation
            if not self.session_state.conversation_id:
                response = await self.api_client.post_async(
                    "/api/experiments/create",
                    json={
                        "mode": self.session_state.chat_mode,
                        "dataset_id": self.session_state.get("dataset_id"),
                    },
                )
                self.session_state.conversation_id = response["id"]

            # Send message
            response = await self.api_client.post_async(
                "/api/chat/message",
                json={
                    "conversation_id": self.session_state.conversation_id,
                    "content": content,
                    "mode": self.session_state.chat_mode,
                    "settings": {
                        "show_thinking": self.session_state.get(
                            "show_agent_thinking", False
                        ),
                        "detail_level": self.session_state.get(
                            "response_detail", "normal"
                        ),
                        "timeout": self.session_state.get("response_timeout", 120),
                    },
                },
            )

            # Handle response
            if response["status"] == "pending_approval":
                self.session_state.pending_approval = response["approval_request"]
            else:
                self._add_assistant_message(response["message"])

            # Trigger callback
            if self.on_message_callback:
                self.on_message_callback(response)

        except Exception as e:
            self._add_system_message(f"Error: {str(e)}", error=True)

        finally:
            self.session_state.is_processing = False

    def _handle_approval(self, approved: bool):
        """
        Handle approval decision.

        Args:
            approved: Whether action was approved
        """
        if not self.session_state.pending_approval:
            return

        feedback = self.session_state.get("approval_feedback", "")

        # Send approval decision
        asyncio.run(self._send_approval_async(approved, feedback))

        # Clear pending approval
        self.session_state.pending_approval = None

    async def _send_approval_async(self, approved: bool, feedback: str):
        """
        Send approval decision to backend.

        Args:
            approved: Whether action was approved
            feedback: User feedback
        """
        try:
            response = await self.api_client.post_async(
                "/api/chat/approval",
                json={
                    "conversation_id": self.session_state.conversation_id,
                    "approved": approved,
                    "feedback": feedback,
                },
            )

            # Add response message
            self._add_assistant_message(response["message"])

        except Exception as e:
            self._add_system_message(f"Error handling approval: {str(e)}", error=True)

    def _add_assistant_message(self, message_data: Dict[str, Any]):
        """Add assistant message from response."""
        message = Message(
            role=message_data.get("role", "assistant"),
            content=message_data["content"],
            timestamp=datetime.now(),
            metadata=message_data.get("metadata"),
            agent_name=message_data.get("agent_name"),
            message_type=message_data.get("type", "text"),
        )
        self.session_state.messages.append(message)

    def _add_system_message(self, content: str, error: bool = False):
        """Add system message."""
        message = Message(
            role="system",
            content=f"⚠️ {content}" if error else content,
            timestamp=datetime.now(),
        )
        self.session_state.messages.append(message)

    def _clear_chat(self):
        """Clear chat history."""
        self.session_state.messages = []
        self.session_state.conversation_id = None
        self.session_state.pending_approval = None
        self._add_system_message("Chat cleared. Start a new conversation!")

    def _get_agent_avatar(self, agent_name: Optional[str]) -> str:
        """Get avatar for agent based on name."""
        agent_avatars = {
            "Supervisor": "👨‍💼",
            "Analysis": "📊",
            "Preprocessing": "🧹",
            "Model": "🤖",
            "Optimization": "⚡",
        }

        if agent_name:
            for key, avatar in agent_avatars.items():
                if key.lower() in agent_name.lower():
                    return avatar

        return "🤖"


# Helper function for creating chat interface
def create_chat_interface(api_client: APIClient, session_state: Any) -> ChatInterface:
    """
    Create and return a chat interface instance.

    Args:
        api_client: API client instance
        session_state: Streamlit session state

    Returns:
        ChatInterface instance
    """
    return ChatInterface(api_client, session_state)
