"""
Session management utilities for AutoML Builder frontend.

This module provides session state management, authentication persistence,
and user preference handling for the Streamlit application.
"""

import streamlit as st
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
import json
import jwt
from functools import wraps
import hashlib
import secrets

from ...core.config import settings


class SessionManager:
    """
    Manages Streamlit session state and user sessions.

    Handles authentication, state persistence, and session lifecycle.
    """

    # Session configuration
    SESSION_TIMEOUT_HOURS = 24
    REMEMBER_ME_DAYS = 30

    # Required session keys
    REQUIRED_KEYS = [
        "authenticated",
        "user_id",
        "user_email",
        "user_name",
        "auth_token",
        "refresh_token",
        "session_id",
        "login_time",
        "last_activity",
    ]

    # Persistent preference keys
    PREFERENCE_KEYS = [
        "theme",
        "chat_mode",
        "debug_mode",
        "show_agent_thinking",
        "response_detail",
        "notification_enabled",
    ]

    def __init__(self):
        """Initialize session manager."""
        self._initialize_session()

    def _initialize_session(self):
        """Initialize session state with default values."""
        # Authentication state
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False

        if "session_id" not in st.session_state:
            st.session_state.session_id = self._generate_session_id()

        # User information
        if "user_id" not in st.session_state:
            st.session_state.user_id = None

        if "user_email" not in st.session_state:
            st.session_state.user_email = None

        if "user_name" not in st.session_state:
            st.session_state.user_name = None

        # Authentication tokens
        if "auth_token" not in st.session_state:
            st.session_state.auth_token = None

        if "refresh_token" not in st.session_state:
            st.session_state.refresh_token = None

        # Session metadata
        if "login_time" not in st.session_state:
            st.session_state.login_time = None

        if "last_activity" not in st.session_state:
            st.session_state.last_activity = datetime.now()

        # User preferences
        if "preferences" not in st.session_state:
            st.session_state.preferences = self._load_default_preferences()

        # Application state
        if "current_page" not in st.session_state:
            st.session_state.current_page = "home"

        if "navigation_history" not in st.session_state:
            st.session_state.navigation_history = []

        # Feature flags
        if "feature_flags" not in st.session_state:
            st.session_state.feature_flags = {}

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return hashlib.sha256(
            f"{secrets.token_urlsafe(32)}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:32]

    def _load_default_preferences(self) -> Dict[str, Any]:
        """Load default user preferences."""
        return {
            "theme": "light",
            "chat_mode": "interactive",
            "debug_mode": False,
            "show_agent_thinking": False,
            "response_detail": "normal",
            "notification_enabled": True,
            "auto_save": True,
            "confirm_actions": True,
            "sidebar_collapsed": False,
            "recent_datasets": [],
            "recent_experiments": [],
        }

    def login(
        self,
        user_id: str,
        user_email: str,
        user_name: str,
        auth_token: str,
        refresh_token: Optional[str] = None,
        remember_me: bool = False,
    ):
        """
        Log in a user and initialize their session.

        Args:
            user_id: User ID
            user_email: User email
            user_name: User display name
            auth_token: JWT authentication token
            refresh_token: Optional refresh token
            remember_me: Whether to extend session duration
        """
        # Set authentication state
        st.session_state.authenticated = True
        st.session_state.user_id = user_id
        st.session_state.user_email = user_email
        st.session_state.user_name = user_name
        st.session_state.auth_token = auth_token
        st.session_state.refresh_token = refresh_token
        st.session_state.login_time = datetime.now()
        st.session_state.last_activity = datetime.now()

        # Set session duration
        if remember_me:
            st.session_state.session_expiry = datetime.now() + timedelta(
                days=self.REMEMBER_ME_DAYS
            )
        else:
            st.session_state.session_expiry = datetime.now() + timedelta(
                hours=self.SESSION_TIMEOUT_HOURS
            )

        # Load user preferences
        self._load_user_preferences(user_id)

        # Update navigation
        st.session_state.current_page = "chat"

        # Log session creation
        self._log_session_event(
            "login", {"user_id": user_id, "remember_me": remember_me}
        )

    def logout(self):
        """Log out the current user and clear session."""
        # Save preferences before logout
        if st.session_state.authenticated:
            self._save_user_preferences()

        # Clear authentication state
        st.session_state.authenticated = False
        st.session_state.user_id = None
        st.session_state.user_email = None
        st.session_state.user_name = None
        st.session_state.auth_token = None
        st.session_state.refresh_token = None
        st.session_state.login_time = None

        # Clear application state
        st.session_state.current_dataset = None
        st.session_state.dataset_id = None
        st.session_state.conversation_id = None
        st.session_state.messages = []
        st.session_state.uploaded_datasets = []

        # Reset to home page
        st.session_state.current_page = "home"

        # Log session end
        self._log_session_event("logout", {})

        # Generate new session ID
        st.session_state.session_id = self._generate_session_id()

    def is_authenticated(self) -> bool:
        """
        Check if user is authenticated and session is valid.

        Returns:
            True if authenticated and session valid, False otherwise
        """
        if not st.session_state.authenticated:
            return False

        # Check session expiry
        if hasattr(st.session_state, "session_expiry"):
            if datetime.now() > st.session_state.session_expiry:
                self.logout()
                return False

        # Validate token
        if not self._validate_token(st.session_state.auth_token):
            self.logout()
            return False

        # Update last activity
        st.session_state.last_activity = datetime.now()

        return True

    def _validate_token(self, token: Optional[str]) -> bool:
        """
        Validate JWT token.

        Args:
            token: JWT token to validate

        Returns:
            True if valid, False otherwise
        """
        if not token:
            return False

        try:
            # Decode and validate token
            payload = jwt.decode(
                token, settings.api.secret_key, algorithms=[settings.api.jwt_algorithm]
            )

            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp) < datetime.now():
                return False

            return True

        except jwt.InvalidTokenError:
            return False

    def refresh_session(self) -> bool:
        """
        Refresh the user session using refresh token.

        Returns:
            True if refresh successful, False otherwise
        """
        if not st.session_state.refresh_token:
            return False

        try:
            # TODO: Call API to refresh token
            # For now, just extend session
            st.session_state.last_activity = datetime.now()

            if hasattr(st.session_state, "session_expiry"):
                st.session_state.session_expiry = datetime.now() + timedelta(
                    hours=self.SESSION_TIMEOUT_HOURS
                )

            return True

        except Exception:
            return False

    def get_user_info(self) -> Dict[str, Any]:
        """
        Get current user information.

        Returns:
            Dictionary containing user info
        """
        if not self.is_authenticated():
            return {}

        return {
            "id": st.session_state.user_id,
            "email": st.session_state.user_email,
            "name": st.session_state.user_name,
            "session_id": st.session_state.session_id,
            "login_time": st.session_state.login_time,
            "last_activity": st.session_state.last_activity,
        }

    def get_preference(self, key: str, default: Any = None) -> Any:
        """
        Get user preference value.

        Args:
            key: Preference key
            default: Default value if not found

        Returns:
            Preference value or default
        """
        return st.session_state.preferences.get(key, default)

    def set_preference(self, key: str, value: Any):
        """
        Set user preference value.

        Args:
            key: Preference key
            value: Preference value
        """
        st.session_state.preferences[key] = value

        # Auto-save if enabled
        if self.get_preference("auto_save", True):
            self._save_user_preferences()

    def update_preferences(self, preferences: Dict[str, Any]):
        """
        Update multiple preferences at once.

        Args:
            preferences: Dictionary of preferences to update
        """
        st.session_state.preferences.update(preferences)

        if self.get_preference("auto_save", True):
            self._save_user_preferences()

    def _load_user_preferences(self, user_id: str):
        """
        Load user preferences from backend.

        Args:
            user_id: User ID
        """
        try:
            # TODO: Implement API call to load preferences
            # For now, use defaults
            pass
        except Exception:
            pass

    def _save_user_preferences(self):
        """Save user preferences to backend."""
        if not self.is_authenticated():
            return

        try:
            # TODO: Implement API call to save preferences
            pass
        except Exception:
            pass

    def navigate_to(self, page: str):
        """
        Navigate to a different page.

        Args:
            page: Page name to navigate to
        """
        # Add to navigation history
        if st.session_state.current_page != page:
            st.session_state.navigation_history.append(st.session_state.current_page)

            # Limit history size
            if len(st.session_state.navigation_history) > 10:
                st.session_state.navigation_history.pop(0)

        st.session_state.current_page = page

    def go_back(self):
        """Navigate to previous page."""
        if st.session_state.navigation_history:
            previous_page = st.session_state.navigation_history.pop()
            st.session_state.current_page = previous_page

    def add_recent_dataset(self, dataset_id: str, dataset_name: str):
        """
        Add dataset to recent datasets list.

        Args:
            dataset_id: Dataset ID
            dataset_name: Dataset name
        """
        recent = self.get_preference("recent_datasets", [])

        # Remove if already exists
        recent = [d for d in recent if d["id"] != dataset_id]

        # Add to front
        recent.insert(
            0,
            {
                "id": dataset_id,
                "name": dataset_name,
                "timestamp": datetime.now().isoformat(),
            },
        )

        # Limit to 10 recent datasets
        recent = recent[:10]

        self.set_preference("recent_datasets", recent)

    def add_recent_experiment(self, experiment_id: str, experiment_name: str):
        """
        Add experiment to recent experiments list.

        Args:
            experiment_id: Experiment ID
            experiment_name: Experiment name
        """
        recent = self.get_preference("recent_experiments", [])

        # Remove if already exists
        recent = [e for e in recent if e["id"] != experiment_id]

        # Add to front
        recent.insert(
            0,
            {
                "id": experiment_id,
                "name": experiment_name,
                "timestamp": datetime.now().isoformat(),
            },
        )

        # Limit to 10 recent experiments
        recent = recent[:10]

        self.set_preference("recent_experiments", recent)

    def get_feature_flag(self, flag_name: str, default: bool = False) -> bool:
        """
        Get feature flag value.

        Args:
            flag_name: Feature flag name
            default: Default value if not found

        Returns:
            Feature flag value
        """
        return st.session_state.feature_flags.get(flag_name, default)

    def set_feature_flags(self, flags: Dict[str, bool]):
        """
        Set feature flags.

        Args:
            flags: Dictionary of feature flags
        """
        st.session_state.feature_flags.update(flags)

    def _log_session_event(self, event_type: str, data: Dict[str, Any]):
        """
        Log session event for analytics.

        Args:
            event_type: Type of event
            data: Event data
        """
        # TODO: Implement event logging
        pass

    def clear_state(self, preserve_auth: bool = True):
        """
        Clear session state while optionally preserving authentication.

        Args:
            preserve_auth: Whether to preserve authentication state
        """
        # Save auth state if needed
        auth_state = {}
        if preserve_auth and self.is_authenticated():
            for key in self.REQUIRED_KEYS:
                if hasattr(st.session_state, key):
                    auth_state[key] = getattr(st.session_state, key)

        # Save preferences
        preferences = st.session_state.preferences.copy()

        # Clear all state
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        # Re-initialize
        self._initialize_session()

        # Restore auth state
        if auth_state:
            for key, value in auth_state.items():
                setattr(st.session_state, key, value)

        # Restore preferences
        st.session_state.preferences = preferences


# Decorator for requiring authentication
def require_auth(func: Callable) -> Callable:
    """
    Decorator to require authentication for a function.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        session = SessionManager()
        if not session.is_authenticated():
            st.error("🔒 Please log in to access this feature.")
            st.stop()
        return func(*args, **kwargs)

    return wrapper


# Decorator for tracking page views
def track_page(page_name: str) -> Callable:
    """
    Decorator to track page views.

    Args:
        page_name: Name of the page

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            session = SessionManager()
            session.navigate_to(page_name)
            return func(*args, **kwargs)

        return wrapper

    return decorator


# Global session instance
session_manager = SessionManager()


# Convenience functions
def get_session() -> SessionManager:
    """Get the global session manager instance."""
    return session_manager


def is_authenticated() -> bool:
    """Check if user is authenticated."""
    return session_manager.is_authenticated()


def get_user_preference(key: str, default: Any = None) -> Any:
    """Get user preference value."""
    return session_manager.get_preference(key, default)


def set_user_preference(key: str, value: Any):
    """Set user preference value."""
    session_manager.set_preference(key, value)


# Export public API
__all__ = [
    "SessionManager",
    "session_manager",
    "get_session",
    "is_authenticated",
    "get_user_preference",
    "set_user_preference",
    "require_auth",
    "track_page",
]
