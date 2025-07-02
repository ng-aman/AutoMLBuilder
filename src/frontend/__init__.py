"""
AutoML Builder Frontend Package.

This package contains the Streamlit-based frontend components for the AutoML Builder
application, including the chat interface, file uploader, debug console, and various
utilities for session management and API communication.
"""

# Version information
__version__ = "1.0.0"
__author__ = "AutoML Builder Team"
__description__ = "Interactive frontend for AutoML multi-agent system"

# Import main components
from .components.chat_interface import ChatInterface, create_chat_interface
from .components.debug_console import DebugConsole, create_debug_console
from .components.file_uploader import FileUploader, create_file_uploader
from .components.auth import AuthComponent, create_auth_component

# Import utilities
from .utils.session import (
    SessionManager,
    session_manager,
    get_session,
    is_authenticated,
    get_user_preference,
    set_user_preference,
    require_auth,
    track_page,
)
from .utils.api_client import APIClient, create_api_client

# Import page modules
from .pages import home, chat, experiments, debug

# Configuration and constants
FRONTEND_CONFIG = {
    "app_name": "AutoML Builder",
    "app_icon": "🤖",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "menu_items": {
        "Get Help": "https://github.com/automl-builder/docs",
        "Report a bug": "https://github.com/automl-builder/issues",
        "About": "AutoML Builder - Intelligent Machine Learning Automation",
    },
}

# Streamlit page configuration
PAGE_CONFIG = {
    "page_title": "AutoML Builder",
    "page_icon": "🤖",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "menu_items": FRONTEND_CONFIG["menu_items"],
}

# Navigation configuration
NAVIGATION_PAGES = {
    "home": {
        "title": "🏠 Home",
        "module": home,
        "requires_auth": False,
        "description": "Welcome to AutoML Builder",
    },
    "chat": {
        "title": "💬 Chat Assistant",
        "module": chat,
        "requires_auth": True,
        "description": "Interactive ML assistant",
    },
    "experiments": {
        "title": "🧪 Experiments",
        "module": experiments,
        "requires_auth": True,
        "description": "View and manage ML experiments",
    },
    "debug": {
        "title": "🐛 Debug Console",
        "module": debug,
        "requires_auth": True,
        "description": "Real-time system monitoring",
    },
}

# Theme configuration
THEME_CONFIG = {
    "light": {
        "primaryColor": "#1E88E5",
        "backgroundColor": "#FFFFFF",
        "secondaryBackgroundColor": "#F0F2F6",
        "textColor": "#262730",
        "font": "sans serif",
    },
    "dark": {
        "primaryColor": "#1E88E5",
        "backgroundColor": "#0E1117",
        "secondaryBackgroundColor": "#262730",
        "textColor": "#FAFAFA",
        "font": "sans serif",
    },
}

# API endpoints configuration
API_ENDPOINTS = {
    "base_url": "http://localhost:8000",
    "auth": {
        "login": "/api/auth/login",
        "logout": "/api/auth/logout",
        "refresh": "/api/auth/refresh",
        "profile": "/api/auth/profile",
    },
    "datasets": {
        "upload": "/api/datasets/upload",
        "list": "/api/datasets",
        "get": "/api/datasets/{id}",
        "delete": "/api/datasets/{id}",
        "preview": "/api/datasets/{id}/preview",
    },
    "chat": {
        "message": "/api/chat/message",
        "history": "/api/chat/history",
        "approval": "/api/chat/approval",
    },
    "experiments": {
        "create": "/api/experiments/create",
        "list": "/api/experiments",
        "get": "/api/experiments/{id}",
        "delete": "/api/experiments/{id}",
        "results": "/api/experiments/{id}/results",
    },
    "debug": {"stream": "/api/debug/stream", "events": "/api/debug/events"},
}

# WebSocket configuration
WEBSOCKET_CONFIG = {
    "debug_stream": "ws://localhost:8000/api/debug/stream",
    "chat_stream": "ws://localhost:8000/api/chat/stream",
    "reconnect_interval": 5,  # seconds
    "max_reconnect_attempts": 10,
}

# File upload configuration
FILE_UPLOAD_CONFIG = {
    "max_size_mb": 100,
    "allowed_extensions": [".csv", ".xlsx", ".xls", ".json", ".parquet"],
    "chunk_size": 1024 * 1024,  # 1MB chunks
    "timeout": 300,  # seconds
}

# Chat configuration
CHAT_CONFIG = {
    "max_message_length": 2000,
    "typing_delay": 50,  # milliseconds
    "auto_scroll": True,
    "show_timestamps": True,
    "enable_markdown": True,
    "enable_code_highlighting": True,
    "quick_actions": [
        {
            "label": "📊 Analyze Data",
            "prompt": "Analyze my dataset and show me insights",
        },
        {"label": "🧹 Preprocess", "prompt": "Clean and preprocess my data"},
        {"label": "🤖 Train Models", "prompt": "Train and compare different models"},
        {
            "label": "⚡ Optimize",
            "prompt": "Optimize hyperparameters for the best model",
        },
    ],
}

# Debug console configuration
DEBUG_CONFIG = {
    "max_events": 1000,
    "default_filters": {
        "levels": ["info", "warning", "error"],
        "sources": [],
        "search": "",
    },
    "event_retention": 3600,  # seconds
    "auto_pause_on_error": True,
}


# Initialize frontend module
def initialize_frontend():
    """
    Initialize the frontend module.

    This function is called when the frontend module is imported and performs
    any necessary initialization tasks.
    """
    import streamlit as st

    # Configure Streamlit page
    if not st.get_option("browser.gatherUsageStats"):
        st.set_page_config(**PAGE_CONFIG)

    # Initialize session manager
    session = get_session()

    # Load feature flags
    try:
        # TODO: Load feature flags from backend
        pass
    except Exception:
        pass

    return True


# Initialize on import
_initialized = False
if not _initialized:
    try:
        _initialized = initialize_frontend()
    except Exception:
        # Silently fail if Streamlit is not available (e.g., during testing)
        _initialized = True


# Helper functions
def get_navigation_pages(authenticated: bool = False) -> dict:
    """
    Get navigation pages based on authentication status.

    Args:
        authenticated: Whether user is authenticated

    Returns:
        Dictionary of available navigation pages
    """
    if authenticated:
        return NAVIGATION_PAGES
    else:
        return {
            key: page
            for key, page in NAVIGATION_PAGES.items()
            if not page["requires_auth"]
        }


def get_api_endpoint(category: str, endpoint: str, **kwargs) -> str:
    """
    Get formatted API endpoint URL.

    Args:
        category: Endpoint category (e.g., "auth", "datasets")
        endpoint: Endpoint name (e.g., "login", "upload")
        **kwargs: Parameters to format in the URL

    Returns:
        Formatted endpoint URL
    """
    base_url = API_ENDPOINTS["base_url"]
    endpoint_path = API_ENDPOINTS.get(category, {}).get(endpoint, "")

    # Format parameters
    if kwargs:
        endpoint_path = endpoint_path.format(**kwargs)

    return f"{base_url}{endpoint_path}"


def configure_theme(theme: str = "light"):
    """
    Configure application theme.

    Args:
        theme: Theme name ("light" or "dark")
    """
    import streamlit as st

    theme_config = THEME_CONFIG.get(theme, THEME_CONFIG["light"])

    # Apply theme using CSS
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-color: {theme_config['backgroundColor']};
        color: {theme_config['textColor']};
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )


def show_notification(message: str, type: str = "info"):
    """
    Show a notification message.

    Args:
        message: Notification message
        type: Notification type ("info", "success", "warning", "error")
    """
    import streamlit as st

    if type == "success":
        st.success(message)
    elif type == "warning":
        st.warning(message)
    elif type == "error":
        st.error(message)
    else:
        st.info(message)


# Export all public components and utilities
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__description__",
    # Components
    "ChatInterface",
    "create_chat_interface",
    "DebugConsole",
    "create_debug_console",
    "FileUploader",
    "create_file_uploader",
    "AuthComponent",
    "create_auth_component",
    # Utilities
    "SessionManager",
    "session_manager",
    "get_session",
    "is_authenticated",
    "get_user_preference",
    "set_user_preference",
    "require_auth",
    "track_page",
    "APIClient",
    "create_api_client",
    # Configuration
    "FRONTEND_CONFIG",
    "PAGE_CONFIG",
    "NAVIGATION_PAGES",
    "THEME_CONFIG",
    "API_ENDPOINTS",
    "WEBSOCKET_CONFIG",
    "FILE_UPLOAD_CONFIG",
    "CHAT_CONFIG",
    "DEBUG_CONFIG",
    # Helper functions
    "initialize_frontend",
    "get_navigation_pages",
    "get_api_endpoint",
    "configure_theme",
    "show_notification",
]
