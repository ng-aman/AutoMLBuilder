# src/frontend/components/auth.py
import streamlit as st
from typing import Dict, Any, Optional
import re


class AuthComponent:
    """Component for handling user authentication"""

    def __init__(self, api_client):
        self.api_client = api_client

    def render_login(self):
        """Render login page"""
        # Center the login form
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.title("🤖 AutoML Builder")
            st.markdown("### Sign in to continue")

            # Login tabs
            tab1, tab2 = st.tabs(["Email Login", "Social Login"])

            with tab1:
                self._render_email_login()

            with tab2:
                self._render_social_login()

            # Demo mode info
            st.info("🎯 **Demo Mode**: Use any email to login (no password required)")

            # Footer
            st.markdown("---")
            st.caption(
                "By signing in, you agree to our Terms of Service and Privacy Policy"
            )

    def _render_email_login(self):
        """Render email login form"""
        with st.form("login_form"):
            email = st.text_input(
                "Email", placeholder="your@email.com", key="login_email"
            )

            # Password field (not used in demo mode)
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter password",
                key="login_password",
                help="Not required in demo mode",
            )

            col1, col2 = st.columns(2)

            with col1:
                remember_me = st.checkbox("Remember me")

            with col2:
                st.markdown("[Forgot password?](#)")

            submit = st.form_submit_button(
                "Sign In", use_container_width=True, type="primary"
            )

            if submit:
                if self._validate_email(email):
                    self._handle_email_login(email, password)
                else:
                    st.error("Please enter a valid email address")

    def _render_social_login(self):
        """Render social login options"""
        st.markdown("Sign in with your social account")

        # Google login
        if st.button(
            "🔵 Sign in with Google", use_container_width=True, key="google_login"
        ):
            self._handle_social_login("google")

        # GitHub login
        if st.button(
            "⚫ Sign in with GitHub", use_container_width=True, key="github_login"
        ):
            self._handle_social_login("github")

        # More providers can be added here
        st.caption("More login options coming soon...")

    def _validate_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return re.match(pattern, email) is not None

    def _handle_email_login(self, email: str, password: Optional[str] = None):
        """Handle email login"""
        with st.spinner("Signing in..."):
            response = self.api_client.login(email, password)

            if response:
                # Store authentication data
                st.session_state.authenticated = True
                st.session_state.user = response["user"]
                st.session_state.access_token = response["access_token"]

                # Get session manager and update
                if "session_manager" in st.session_state:
                    st.session_state.session_manager.set_user(
                        response["user"], response["access_token"]
                    )

                st.success(f"Welcome back, {response['user'].get('name', email)}!")
                st.rerun()
            else:
                st.error("Login failed. Please try again.")

    def _handle_social_login(self, provider: str):
        """Handle social login"""
        # Get OAuth URL
        redirect_uri = st.get_url() + f"?oauth_callback={provider}"

        response = self.api_client.get_oauth_url(provider, redirect_uri)

        if response and response.get("authorization_url"):
            # Store state for verification
            st.session_state[f"oauth_state_{provider}"] = response["state"]

            # Redirect to OAuth provider
            st.markdown(
                f'<meta http-equiv="refresh" content="0; url={response["authorization_url"]}">',
                unsafe_allow_html=True,
            )

            st.info(f"Redirecting to {provider.title()} login...")

            # Show manual link as backup
            st.markdown(
                f"If you're not redirected, [click here]({response['authorization_url']})"
            )
        else:
            st.error(f"Failed to initialize {provider.title()} login")

    def handle_oauth_callback(self, provider: str, code: str, state: str):
        """Handle OAuth callback"""
        # Verify state
        stored_state = st.session_state.get(f"oauth_state_{provider}")
        if not stored_state or stored_state != state:
            st.error("Invalid OAuth state. Please try logging in again.")
            return

        # Exchange code for token
        redirect_uri = st.get_url().split("?")[0] + f"?oauth_callback={provider}"

        response = self.api_client.oauth_callback(provider, code, state, redirect_uri)

        if response:
            # Store authentication data
            st.session_state.authenticated = True
            st.session_state.user = response["user"]
            st.session_state.access_token = response["access_token"]

            # Get session manager and update
            if "session_manager" in st.session_state:
                st.session_state.session_manager.set_user(
                    response["user"], response["access_token"]
                )

            # Clean up OAuth state
            if f"oauth_state_{provider}" in st.session_state:
                del st.session_state[f"oauth_state_{provider}"]

            st.success(f"Successfully logged in with {provider.title()}!")
            st.rerun()
        else:
            st.error(f"{provider.title()} login failed. Please try again.")

    def logout(self):
        """Handle user logout"""
        # Call logout API
        self.api_client.logout()

        # Clear session state
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.access_token = None

        # Clear session manager
        if "session_manager" in st.session_state:
            st.session_state.session_manager.clear_user()

        st.success("Logged out successfully")
        st.rerun()

    def render_user_menu(self):
        """Render user menu in sidebar"""
        if st.session_state.get("user"):
            user = st.session_state.user

            st.sidebar.markdown("---")
            st.sidebar.markdown("### 👤 Account")

            # User info
            st.sidebar.write(f"**{user.get('name', 'User')}**")
            st.sidebar.caption(user.get("email", ""))

            # Account actions
            if st.sidebar.button("⚙️ Settings", use_container_width=True):
                st.session_state.show_settings = True

            if st.sidebar.button("🚪 Logout", use_container_width=True):
                self.logout()

    def require_auth(self, page_func):
        """Decorator to require authentication for a page"""

        def wrapper(*args, **kwargs):
            if not st.session_state.get("authenticated"):
                st.warning("Please login to access this page")
                self.render_login()
                return
            return page_func(*args, **kwargs)

        return wrapper


# OAuth callback handler
def check_oauth_callback():
    """Check if current request is an OAuth callback"""
    query_params = st.query_params

    # Check for OAuth callback parameters
    for provider in ["google", "github"]:
        if query_params.get(f"oauth_callback") == provider:
            code = query_params.get("code")
            state = query_params.get("state")

            if code and state:
                return {"provider": provider, "code": code, "state": state}

    return None
