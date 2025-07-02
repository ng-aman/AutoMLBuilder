# src/api/dependencies/auth.py
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2AuthorizationCodeBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from src.api.dependencies.database import get_db
from src.api.models.user import User
from src.core.config import settings
from src.utils.logger import get_logger
from src.utils.exceptions import AuthenticationError, AuthorizationError

logger = get_logger(__name__)

# OAuth2 schemes
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token", auto_error=False)

# OAuth2 providers configuration
oauth2_providers = {
    "google": OAuth2AuthorizationCodeBearer(
        authorizationUrl="https://accounts.google.com/o/oauth2/v2/auth",
        tokenUrl="https://oauth2.googleapis.com/token",
        auto_error=False,
    ),
    "github": OAuth2AuthorizationCodeBearer(
        authorizationUrl="https://github.com/login/oauth/authorize",
        tokenUrl="https://github.com/login/oauth/access_token",
        auto_error=False,
    ),
}


def create_access_token(
    data: Dict[str, Any], expires_delta: Optional[timedelta] = None
) -> str:
    """Create JWT access token"""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=settings.jwt_expiration_hours)

    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(
        to_encode, settings.api_secret_key, algorithm=settings.jwt_algorithm
    )

    return encoded_jwt


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode and validate JWT token"""
    try:
        payload = jwt.decode(
            token, settings.api_secret_key, algorithms=[settings.jwt_algorithm]
        )
        return payload
    except JWTError as e:
        logger.error("JWT decode error", error=str(e))
        raise AuthenticationError("Invalid token")


async def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user"""
    if not token:
        raise AuthenticationError("No authentication token provided")

    try:
        # Decode token
        payload = decode_access_token(token)
        user_id = payload.get("sub")

        if not user_id:
            raise AuthenticationError("Invalid token payload")

        # Get user from database
        user = db.query(User).filter(User.id == user_id).first()

        if not user:
            raise AuthenticationError("User not found")

        # Update last login
        user.last_login = datetime.utcnow()
        db.commit()

        return user

    except AuthenticationError:
        raise
    except Exception as e:
        logger.error("Authentication error", error=str(e))
        raise AuthenticationError("Authentication failed")


async def get_current_user_optional(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> Optional[User]:
    """Get current user if authenticated, otherwise None"""
    if not token:
        return None

    try:
        return await get_current_user(token, db)
    except:
        return None


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Require admin privileges"""
    # For now, we'll check if email ends with @admin.com
    # In production, you'd have proper role management
    if not current_user.email.endswith("@admin.com"):
        raise AuthorizationError("Admin privileges required")

    return current_user


class OAuth2Handler:
    """Handle OAuth2 authentication flow"""

    def __init__(self, provider: str):
        self.provider = provider
        self.config = settings.get_oauth_providers().get(provider)

        if not self.config:
            raise ValueError(f"OAuth provider {provider} not configured")

    def get_authorization_url(self, redirect_uri: str, state: str) -> str:
        """Get OAuth2 authorization URL"""
        if self.provider == "google":
            return (
                f"https://accounts.google.com/o/oauth2/v2/auth?"
                f"client_id={self.config['client_id']}&"
                f"redirect_uri={redirect_uri}&"
                f"response_type=code&"
                f"scope=openid%20email%20profile&"
                f"state={state}"
            )
        elif self.provider == "github":
            return (
                f"https://github.com/login/oauth/authorize?"
                f"client_id={self.config['client_id']}&"
                f"redirect_uri={redirect_uri}&"
                f"scope=user:email&"
                f"state={state}"
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    async def exchange_code_for_token(
        self, code: str, redirect_uri: str
    ) -> Dict[str, Any]:
        """Exchange authorization code for access token"""
        import httpx

        if self.provider == "google":
            token_url = "https://oauth2.googleapis.com/token"
            data = {
                "code": code,
                "client_id": self.config["client_id"],
                "client_secret": self.config["client_secret"],
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            }
        elif self.provider == "github":
            token_url = "https://github.com/login/oauth/access_token"
            data = {
                "code": code,
                "client_id": self.config["client_id"],
                "client_secret": self.config["client_secret"],
                "redirect_uri": redirect_uri,
            }
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                token_url, data=data, headers={"Accept": "application/json"}
            )

            if response.status_code != 200:
                raise AuthenticationError("Failed to exchange code for token")

            return response.json()

    async def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get user info from OAuth provider"""
        import httpx

        if self.provider == "google":
            user_info_url = "https://www.googleapis.com/oauth2/v2/userinfo"
            headers = {"Authorization": f"Bearer {access_token}"}
        elif self.provider == "github":
            user_info_url = "https://api.github.com/user"
            headers = {"Authorization": f"token {access_token}"}
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        async with httpx.AsyncClient() as client:
            response = await client.get(user_info_url, headers=headers)

            if response.status_code != 200:
                raise AuthenticationError("Failed to get user info")

            user_data = response.json()

            # Get email for GitHub (separate endpoint)
            if self.provider == "github" and not user_data.get("email"):
                email_response = await client.get(
                    "https://api.github.com/user/emails", headers=headers
                )
                if email_response.status_code == 200:
                    emails = email_response.json()
                    primary_email = next(
                        (e["email"] for e in emails if e["primary"]), None
                    )
                    if primary_email:
                        user_data["email"] = primary_email

            return user_data


def create_or_update_oauth_user(
    db: Session, provider: str, oauth_id: str, email: str, name: Optional[str] = None
) -> User:
    """Create or update user from OAuth info"""
    # Check if user exists
    user = db.query(User).filter(User.email == email).first()

    if not user:
        # Create new user
        user = User(email=email, name=name, oauth_provider=provider, oauth_id=oauth_id)
        db.add(user)
        logger.info("Created new OAuth user", email=email, provider=provider)
    else:
        # Update existing user
        user.oauth_provider = provider
        user.oauth_id = oauth_id
        if name:
            user.name = name
        logger.info("Updated existing OAuth user", email=email, provider=provider)

    user.last_login = datetime.utcnow()
    db.commit()
    db.refresh(user)

    return user
