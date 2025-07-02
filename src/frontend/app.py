# src/frontend/app.py
import streamlit as st
from streamlit_option_menu import option_menu
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.frontend.pages import home, chat, experiments, debug
from src.frontend.components.auth import AuthComponent
from src.frontend.utils.session import SessionManager
from src.frontend.utils.api_client import APIClient
from src.core.config import settings

# Page configuration
st.set_page_config(
    page_title="AutoML Builder",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    /* Main container */
    .main {
        padding: 1rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f0f2f6;
    }
    
    /* Custom button styling */
    .stButton > button {
        width: 100%;
        border-radius: 5px;
        border: 1px solid #ddd;
        background-color: white;
        color: #333;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #f0f2f6;
        border-color: #333;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    
    .chat-message.user {
        background-color: #e3f2fd;
    }
    
    .chat-message.assistant {
        background-color: #f5f5f5;
    }
    
    /* Custom metric cards */
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Progress indicators */
    .progress-step {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    
    .progress-step.completed {
        color: #4caf50;
    }
    
    .progress-step.active {
        color: #2196f3;
        font-weight: bold;
    }
    
    .progress-step.pending {
        color: #9e9e9e;
    }
</style>
""",
    unsafe_allow_html=True,
)


def initialize_session_state():
    """Initialize session state variables"""
    if "session_manager" not in st.session_state:
        st.session_state.session_manager = SessionManager()

    if "api_client" not in st.session_state:
        st.session_state.api_client = APIClient(
            base_url=os.getenv("API_URL", "http://localhost:8000")
        )

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if "user" not in st.session_state:
        st.session_state.user = None

    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"


def main():
    """Main application entry point"""
    # Initialize session state
    initialize_session_state()

    # Check authentication
    auth_component = AuthComponent(st.session_state.api_client)

    if not st.session_state.authenticated:
        # Show login page
        auth_component.render_login()
        return

    # Sidebar navigation
    with st.sidebar:
        st.title("🤖 AutoML Builder")

        # User info
        if st.session_state.user:
            st.write(
                f"👤 {st.session_state.user.get('name', st.session_state.user.get('email'))}"
            )
            if st.button("Logout", key="logout_btn"):
                auth_component.logout()
                st.rerun()

        st.divider()

        # Navigation menu
        selected = option_menu(
            menu_title="Navigation",
            options=["Home", "Chat", "Experiments", "Debug"],
            icons=["house", "chat-dots", "graph-up", "bug"],
            menu_icon="cast",
            default_index=0,
            key="navigation",
        )

        st.session_state.current_page = selected

        # Settings section
        st.divider()
        st.subheader("⚙️ Settings")

        # Mode selection
        mode = st.radio(
            "Workflow Mode",
            ["🤖 Auto", "🤝 Interactive"],
            index=0 if settings.enable_auto_mode else 1,
            help="Auto: AI handles everything\nInteractive: You approve each step",
        )
        st.session_state.workflow_mode = "auto" if "Auto" in mode else "interactive"

        # Debug mode toggle
        debug_mode = st.checkbox(
            "Debug Mode",
            value=settings.enable_debug_mode,
            help="Show detailed logs and agent decisions",
        )
        st.session_state.debug_mode = debug_mode

        # API Status
        st.divider()
        api_status = st.session_state.api_client.check_health()
        if api_status:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Disconnected")
            if st.button("Retry Connection"):
                st.rerun()

    # Main content area
    if selected == "Home":
        home.render(st.session_state)
    elif selected == "Chat":
        chat.render(st.session_state)
    elif selected == "Experiments":
        experiments.render(st.session_state)
    elif selected == "Debug":
        if st.session_state.debug_mode:
            debug.render(st.session_state)
        else:
            st.warning("Enable Debug Mode in settings to access this page")

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("🤖 Powered by LangGraph & OpenAI")
    with col2:
        st.caption("📊 Experiments tracked with MLflow")
    with col3:
        st.caption("🚀 Built with Streamlit")


if __name__ == "__main__":
    main()
