# src/frontend/utils/api_client.py
import requests
from typing import Dict, Any, List, Optional, BinaryIO
import streamlit as st
from datetime import datetime
import json


class APIClient:
    """Client for interacting with the AutoML API"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def _get_headers(self) -> Dict[str, str]:
        """Get headers with authentication token"""
        headers = {"Content-Type": "application/json"}

        if hasattr(st.session_state, "access_token") and st.session_state.access_token:
            headers["Authorization"] = f"Bearer {st.session_state.access_token}"

        return headers

    def _handle_response(self, response: requests.Response) -> Optional[Dict[str, Any]]:
        """Handle API response"""
        try:
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                st.error("Authentication failed. Please login again.")
                st.session_state.authenticated = False
                return None
            elif response.status_code == 404:
                st.error("Resource not found")
                return None
            else:
                error_data = response.json()
                st.error(f"API Error: {error_data.get('detail', 'Unknown error')}")
                return None
        except Exception as e:
            st.error(f"Error processing response: {str(e)}")
            return None

    # Health check
    def check_health(self) -> bool:
        """Check if API is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    # Authentication endpoints
    def login(
        self, email: str, password: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Login with email"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/auth/login",
                json={"email": email, "password": password},
            )
            return self._handle_response(response)
        except Exception as e:
            st.error(f"Login failed: {str(e)}")
            return None

    def get_oauth_url(
        self, provider: str, redirect_uri: str
    ) -> Optional[Dict[str, Any]]:
        """Get OAuth authorization URL"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/auth/oauth/{provider}/authorize",
                params={"redirect_uri": redirect_uri},
            )
            return self._handle_response(response)
        except Exception as e:
            st.error(f"OAuth setup failed: {str(e)}")
            return None

    def oauth_callback(
        self, provider: str, code: str, state: str, redirect_uri: str
    ) -> Optional[Dict[str, Any]]:
        """Handle OAuth callback"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/auth/oauth/{provider}/callback",
                json={"code": code, "state": state, "redirect_uri": redirect_uri},
            )
            return self._handle_response(response)
        except Exception as e:
            st.error(f"OAuth callback failed: {str(e)}")
            return None

    def logout(self) -> bool:
        """Logout user"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/auth/logout", headers=self._get_headers()
            )
            return response.status_code == 200
        except:
            return False

    def get_me(self) -> Optional[Dict[str, Any]]:
        """Get current user info"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/auth/me", headers=self._get_headers()
            )
            return self._handle_response(response)
        except Exception as e:
            st.error(f"Failed to get user info: {str(e)}")
            return None

    # Chat endpoints
    def send_message(
        self,
        message: str,
        session_id: Optional[str] = None,
        mode: str = "auto",
        dataset_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Send a chat message"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/chat/message",
                headers=self._get_headers(),
                json={
                    "message": message,
                    "session_id": session_id,
                    "mode": mode,
                    "dataset_id": dataset_id,
                },
            )
            return self._handle_response(response)
        except Exception as e:
            st.error(f"Failed to send message: {str(e)}")
            return None

    def approve_action(
        self, session_id: str, decision: str, reason: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Approve or reject a pending action"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/chat/approve",
                headers=self._get_headers(),
                json={"session_id": session_id, "decision": decision, "reason": reason},
            )
            return self._handle_response(response)
        except Exception as e:
            st.error(f"Failed to submit approval: {str(e)}")
            return None

    def get_chat_sessions(
        self, skip: int = 0, limit: int = 20
    ) -> Optional[Dict[str, Any]]:
        """Get user's chat sessions"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/chat/sessions",
                headers=self._get_headers(),
                params={"skip": skip, "limit": limit},
            )
            return self._handle_response(response)
        except Exception as e:
            st.error(f"Failed to get chat sessions: {str(e)}")
            return None

    def get_session_detail(
        self, session_id: str, include_state: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Get session details with messages"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/chat/session/{session_id}",
                headers=self._get_headers(),
                params={"include_state": include_state},
            )
            return self._handle_response(response)
        except Exception as e:
            st.error(f"Failed to get session details: {str(e)}")
            return None

    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session"""
        try:
            response = self.session.delete(
                f"{self.base_url}/api/chat/session/{session_id}",
                headers=self._get_headers(),
            )
            return response.status_code == 200
        except:
            return False

    # Dataset endpoints
    def upload_dataset(self, file: BinaryIO, filename: str) -> Optional[Dict[str, Any]]:
        """Upload a dataset"""
        try:
            files = {"file": (filename, file, "application/octet-stream")}
            headers = {"Authorization": f"Bearer {st.session_state.access_token}"}

            response = self.session.post(
                f"{self.base_url}/api/datasets/upload", headers=headers, files=files
            )
            return self._handle_response(response)
        except Exception as e:
            st.error(f"Failed to upload dataset: {str(e)}")
            return None

    def get_datasets(self, skip: int = 0, limit: int = 20) -> Optional[Dict[str, Any]]:
        """Get user's datasets"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/datasets",
                headers=self._get_headers(),
                params={"skip": skip, "limit": limit},
            )
            return self._handle_response(response)
        except Exception as e:
            st.error(f"Failed to get datasets: {str(e)}")
            return None

    def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed dataset information"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/datasets/{dataset_id}",
                headers=self._get_headers(),
            )
            return self._handle_response(response)
        except Exception as e:
            st.error(f"Failed to get dataset info: {str(e)}")
            return None

    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset"""
        try:
            response = self.session.delete(
                f"{self.base_url}/api/datasets/{dataset_id}",
                headers=self._get_headers(),
            )
            return response.status_code == 200
        except:
            return False

    def preview_dataset(
        self, dataset_id: str, rows: int = 10
    ) -> Optional[Dict[str, Any]]:
        """Preview dataset contents"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/datasets/{dataset_id}/preview",
                headers=self._get_headers(),
                params={"rows": rows},
            )
            return self._handle_response(response)
        except Exception as e:
            st.error(f"Failed to preview dataset: {str(e)}")
            return None

    # Experiment endpoints
    def get_experiments(
        self,
        session_id: Optional[str] = None,
        status: Optional[str] = None,
        skip: int = 0,
        limit: int = 20,
    ) -> Optional[Dict[str, Any]]:
        """Get experiments"""
        try:
            params = {"skip": skip, "limit": limit}
            if session_id:
                params["session_id"] = session_id
            if status:
                params["status"] = status

            response = self.session.get(
                f"{self.base_url}/api/experiments",
                headers=self._get_headers(),
                params=params,
            )
            return self._handle_response(response)
        except Exception as e:
            st.error(f"Failed to get experiments: {str(e)}")
            return None

    def get_experiment_detail(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed experiment information"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/experiments/{experiment_id}",
                headers=self._get_headers(),
            )
            return self._handle_response(response)
        except Exception as e:
            st.error(f"Failed to get experiment details: {str(e)}")
            return None

    def compare_experiments(
        self, experiment_ids: List[str], metric: str = "accuracy"
    ) -> Optional[Dict[str, Any]]:
        """Compare multiple experiments"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/experiments/compare",
                headers=self._get_headers(),
                json={"experiment_ids": experiment_ids, "metric": metric},
            )
            return self._handle_response(response)
        except Exception as e:
            st.error(f"Failed to compare experiments: {str(e)}")
            return None

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment"""
        try:
            response = self.session.delete(
                f"{self.base_url}/api/experiments/{experiment_id}",
                headers=self._get_headers(),
            )
            return response.status_code == 200
        except:
            return False

    # Debug endpoints
    def get_debug_events(
        self, session_id: str, event_types: Optional[List[str]] = None, limit: int = 100
    ) -> Optional[Dict[str, Any]]:
        """Get debug events for a session"""
        try:
            params = {"limit": limit}
            if event_types:
                params["event_types"] = event_types

            response = self.session.get(
                f"{self.base_url}/api/debug/events/{session_id}",
                headers=self._get_headers(),
                params=params,
            )
            return self._handle_response(response)
        except Exception as e:
            st.error(f"Failed to get debug events: {str(e)}")
            return None

    # WebSocket connection for real-time events
    def get_debug_websocket_url(self, session_id: str) -> str:
        """Get WebSocket URL for debug events"""
        ws_base = self.base_url.replace("http://", "ws://").replace(
            "https://", "wss://"
        )
        return f"{ws_base}/ws/debug/{session_id}"
