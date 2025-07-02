# src/api/__init__.py
"""
AutoML Builder API Module

FastAPI-based REST API for the AutoML Builder platform.
Handles authentication, dataset management, chat interactions, and experiment tracking.
"""

from src.api.main import app

__all__ = ["app"]

__version__ = "1.0.0"
