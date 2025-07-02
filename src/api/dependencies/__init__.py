# src/api/dependencies/__init__.py
"""
FastAPI dependencies for dependency injection.
"""

from .database import get_db, init_database, check_database_connection
from .auth import (
    get_current_user,
    get_current_user_optional,
    require_admin,
    create_access_token,
    OAuth2Handler,
)

__all__ = [
    # Database
    "get_db",
    "init_database",
    "check_database_connection",
    # Authentication
    "get_current_user",
    "get_current_user_optional",
    "require_admin",
    "create_access_token",
    "OAuth2Handler",
]
