# src/frontend/pages/chat.py
import streamlit as st
from datetime import datetime
import requests
from typing import List, Dict, Optional
import json


class ChatInterface:
    def __init__(self, api_url: str):
        self.api_url = api_url
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize session state variables"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "current_session_id" not in st.session_state:
            st.session_state.current_session_id = None
        if "user_token" not in st.session_state:
            st.session_state.user_token = None
        if "chat_sessions" not in st.session_state:
            st.session_state.chat_sessions = []
        if "current_dataset" not in st.session_state:
            st.session_state.current_dataset = None
        if "debug_mode" not in st.session_state:
            st.session_state.debug_mode = False

    def render(self):
        """Main render method for chat interface"""
        # Create layout
        col1, col2 = st.columns([1, 3])

        # Sidebar with chat history
        with col1:
            self._render_sidebar()

        # Main chat interface
        with col2:
            self._render_chat_area()

            # Debug console (if enabled)
            if st.session_state.debug_mode:
                self._render_debug_console()

    def _render_sidebar(self):
        """Render sidebar with chat history"""
        st.header("💬 Chat History")

        # New chat button
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            self._create_new_session()

        st.divider()

        # Mode selection
        mode = st.radio(
            "Mode",
            ["🤖 Auto", "🤝 Interactive"],
            help="Auto: Let AI handle everything\nInteractive: Approve each step",
        )

        # Debug mode toggle
        st.session_state.debug_mode = st.checkbox(
            "🐛 Debug Mode",
            value=st.session_state.debug_mode,
            help="Show LLM calls and agent decisions",
        )

        st.divider()

        # Chat sessions list
        st.subheader("Previous Chats")

        # Fetch chat sessions
        sessions = self._get_chat_sessions()

        # Search box
        search_query = st.text_input("🔍 Search chats...", key="search_chats")

        # Filter sessions
        if search_query:
            sessions = [
                s for s in sessions if search_query.lower() in s["title"].lower()
            ]

        # Display sessions
        for session in sessions:
            container = st.container()
            with container:
                col1, col2 = st.columns([4, 1])

                with col1:
                    if st.button(
                        f"📊 {session['title'][:25]}...",
                        key=f"load_{session['id']}",
                        use_container_width=True,
                    ):
                        self._load_session(session["id"])

                with col2:
                    if st.button("🗑️", key=f"del_{session['id']}"):
                        self._delete_session(session["id"])

                # Show metadata
                st.caption(f"📅 {session['created_at']}")
                if session.get("dataset_name"):
                    st.caption(f"📁 {session['dataset_name']}")

    def _render_chat_area(self):
        """Render main chat area"""
        st.header("🤖 AutoML Assistant")

        # File upload area
        uploaded_file = st.file_uploader(
            "Upload your dataset",
            type=["csv", "xlsx", "json"],
            help="Upload a dataset to start analysis",
        )

        if uploaded_file:
            if uploaded_file != st.session_state.current_dataset:
                self._handle_file_upload(uploaded_file)

        # Chat messages container
        chat_container = st.container()
        with chat_container:
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

                    # Show metadata if in debug mode
                    if st.session_state.debug_mode and message.get("metadata"):
                        with st.expander("🔍 Debug Info"):
                            st.json(message["metadata"])

        # Chat input
        if prompt := st.chat_input("Ask me anything about your data..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = self._send_message(prompt)
                    st.markdown(response["content"])

                    # Add to messages
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response["content"],
                            "metadata": response.get("metadata"),
                        }
                    )

    def _render_debug_console(self):
        """Render debug console for agent events"""
        with st.expander("🐛 Debug Console", expanded=True):
            st.subheader("Agent Events")

            # Connect to WebSocket for real-time events
            if st.session_state.current_session_id:
                events = self._get_debug_events()

                # Display events in reverse chronological order
                for event in reversed(events[-20:]):  # Show last 20 events
                    event_type = event.get("type", "unknown")
                    timestamp = event.get("timestamp", "")

                    # Color code by event type
                    if event_type == "llm_call":
                        st.info(
                            f"🤖 {timestamp}: LLM Call - {event.get('model', 'unknown')}"
                        )
                        with st.expander("Details"):
                            st.json(event.get("details", {}))

                    elif event_type == "agent_decision":
                        st.success(
                            f"🎯 {timestamp}: {event.get('agent', 'unknown')} - {event.get('decision', '')}"
                        )

                    elif event_type == "tool_call":
                        st.warning(
                            f"🔧 {timestamp}: Tool - {event.get('tool', 'unknown')}"
                        )
                        with st.expander("Parameters"):
                            st.json(event.get("parameters", {}))

                    elif event_type == "error":
                        st.error(
                            f"❌ {timestamp}: Error - {event.get('message', 'unknown error')}"
                        )

    def _create_new_session(self):
        """Create a new chat session"""
        st.session_state.messages = []
        st.session_state.current_session_id = None
        st.session_state.current_dataset = None
        st.rerun()

    def _load_session(self, session_id: str):
        """Load a previous chat session"""
        try:
            response = requests.get(
                f"{self.api_url}/chat/session/{session_id}",
                headers={"Authorization": f"Bearer {st.session_state.user_token}"},
            )

            if response.status_code == 200:
                data = response.json()
                st.session_state.messages = data["messages"]
                st.session_state.current_session_id = session_id
                st.success("Session loaded successfully!")
                st.rerun()
            else:
                st.error("Failed to load session")
        except Exception as e:
            st.error(f"Error loading session: {str(e)}")

    def _delete_session(self, session_id: str):
        """Delete a chat session"""
        try:
            response = requests.delete(
                f"{self.api_url}/chat/session/{session_id}",
                headers={"Authorization": f"Bearer {st.session_state.user_token}"},
            )

            if response.status_code == 200:
                st.success("Session deleted")
                if session_id == st.session_state.current_session_id:
                    self._create_new_session()
                else:
                    st.rerun()
            else:
                st.error("Failed to delete session")
        except Exception as e:
            st.error(f"Error deleting session: {str(e)}")

    def _get_chat_sessions(self) -> List[Dict]:
        """Fetch user's chat sessions"""
        try:
            response = requests.get(
                f"{self.api_url}/chat/sessions",
                headers={"Authorization": f"Bearer {st.session_state.user_token}"},
            )

            if response.status_code == 200:
                return response.json()
            else:
                return []
        except Exception as e:
            st.error(f"Error fetching sessions: {str(e)}")
            return []

    def _send_message(self, message: str) -> Dict:
        """Send message to API and get response"""
        try:
            payload = {
                "message": message,
                "session_id": st.session_state.current_session_id,
                "mode": (
                    "auto"
                    if "Auto" in st.session_state.get("mode", "Auto")
                    else "interactive"
                ),
                "dataset_id": st.session_state.current_dataset,
            }

            response = requests.post(
                f"{self.api_url}/chat/message",
                json=payload,
                headers={"Authorization": f"Bearer {st.session_state.user_token}"},
            )

            if response.status_code == 200:
                data = response.json()
                # Update session ID if new session created
                if not st.session_state.current_session_id:
                    st.session_state.current_session_id = data.get("session_id")
                return data
            else:
                return {
                    "content": "Sorry, I encountered an error processing your request.",
                    "metadata": {"error": response.text},
                }
        except Exception as e:
            return {"content": f"Error: {str(e)}", "metadata": {"error": str(e)}}

    def _handle_file_upload(self, uploaded_file):
        """Handle file upload"""
        try:
            # Upload file to API
            files = {"file": uploaded_file}
            response = requests.post(
                f"{self.api_url}/datasets/upload",
                files=files,
                headers={"Authorization": f"Bearer {st.session_state.user_token}"},
            )

            if response.status_code == 200:
                data = response.json()
                st.session_state.current_dataset = data["dataset_id"]
                st.success(f"✅ Uploaded {uploaded_file.name}")

                # Add system message
                st.session_state.messages.append(
                    {
                        "role": "system",
                        "content": f"Dataset '{uploaded_file.name}' uploaded successfully. {data.get('info', '')}",
                    }
                )
            else:
                st.error("Failed to upload file")
        except Exception as e:
            st.error(f"Error uploading file: {str(e)}")

    def _get_debug_events(self) -> List[Dict]:
        """Get debug events for current session"""
        # In production, this would connect to WebSocket
        # For now, returning mock data
        return [
            {
                "type": "llm_call",
                "timestamp": "2024-01-10 10:30:15",
                "model": "gpt-4",
                "details": {"prompt": "Analyze the uploaded dataset...", "tokens": 245},
            },
            {
                "type": "agent_decision",
                "timestamp": "2024-01-10 10:30:16",
                "agent": "Analysis Agent",
                "decision": "Dataset contains 5000 rows, 12 columns. Detected classification problem.",
            },
            {
                "type": "tool_call",
                "timestamp": "2024-01-10 10:30:17",
                "tool": "pandas_profiling",
                "parameters": {"dataset_id": "abc123"},
            },
        ]


# Main app entry point
def main():
    st.set_page_config(page_title="AutoML Chat", page_icon="🤖", layout="wide")

    # Check authentication
    if "user_token" not in st.session_state:
        st.error("Please login first")
        st.stop()

    # Initialize and render chat interface
    chat = ChatInterface(api_url="http://localhost:8000/api")
    chat.render()


if __name__ == "__main__":
    main()
