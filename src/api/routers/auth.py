# src/api/routers/auth.py
from fastapi import APIRouter, Depends, HTTPException, status, Response
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import Optional
import secrets
from src.api.dependencies.database import get_db
from src.api.dependencies.auth import (
    create_access_token,
    get_current_user,
    OAuth2Handler,
    create_or_update_oauth_user,
)
from src.api.models.user import User
from src.core.config import settings
from src.utils.logger import get_logger
from pydantic import BaseModel, EmailStr

logger = get_logger(__name__)

router = APIRouter(prefix="/api/auth")


# Pydantic models for requests/responses
class LoginRequest(BaseModel):
    """Login request model"""

    email: EmailStr
    password: Optional[str] = None  # For demo purposes


class LoginResponse(BaseModel):
    """Login response model"""

    access_token: str
    token_type: str = "bearer"
    user: dict


class OAuthCallbackRequest(BaseModel):
    """OAuth callback request"""

    code: str
    state: str
    redirect_uri: str


class UserResponse(BaseModel):
    """User response model"""

    id: str
    email: str
    name: Optional[str]
    created_at: str


# Temporary in-memory state store for OAuth (use Redis in production)
oauth_states = {}


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """Login with email (demo mode - no password check)"""
    # In production, you would verify password here
    user = db.query(User).filter(User.email == request.email).first()

    if not user:
        # Create user for demo
        user = User(email=request.email, name=request.email.split("@")[0])
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info("Created demo user", email=request.email)

    # Create access token
    access_token = create_access_token({"sub": str(user.id)})

    return LoginResponse(access_token=access_token, user=user.to_dict())


@router.get("/oauth/{provider}/authorize")
async def oauth_authorize(provider: str, redirect_uri: str):
    """Get OAuth authorization URL"""
    try:
        # Validate provider
        if provider not in ["google", "github"]:
            raise HTTPException(
                status_code=400, detail=f"Unsupported OAuth provider: {provider}"
            )

        # Generate state for CSRF protection
        state = secrets.token_urlsafe(32)
        oauth_states[state] = {"provider": provider, "redirect_uri": redirect_uri}

        # Get authorization URL
        oauth_handler = OAuth2Handler(provider)
        auth_url = oauth_handler.get_authorization_url(redirect_uri, state)

        return {"authorization_url": auth_url, "state": state}
    except Exception as e:
        logger.error(f"OAuth authorization error", provider=provider, error=str(e))
        raise HTTPException(status_code=500, detail="OAuth configuration error")


@router.post("/oauth/{provider}/callback", response_model=LoginResponse)
async def oauth_callback(
    provider: str, request: OAuthCallbackRequest, db: Session = Depends(get_db)
):
    """Handle OAuth callback"""
    try:
        # Verify state
        if request.state not in oauth_states:
            raise HTTPException(status_code=400, detail="Invalid state")

        state_data = oauth_states.pop(request.state)
        if state_data["provider"] != provider:
            raise HTTPException(status_code=400, detail="Provider mismatch")

        # Exchange code for token
        oauth_handler = OAuth2Handler(provider)
        token_data = await oauth_handler.exchange_code_for_token(
            request.code, request.redirect_uri
        )

        # Get user info
        access_token = token_data.get("access_token")
        if not access_token:
            raise HTTPException(status_code=400, detail="No access token received")

        user_info = await oauth_handler.get_user_info(access_token)

        # Extract user data
        if provider == "google":
            email = user_info.get("email")
            name = user_info.get("name")
            oauth_id = user_info.get("id")
        elif provider == "github":
            email = user_info.get("email")
            name = user_info.get("name") or user_info.get("login")
            oauth_id = str(user_info.get("id"))
        else:
            raise ValueError(f"Unknown provider: {provider}")

        if not email:
            raise HTTPException(
                status_code=400, detail="Email not provided by OAuth provider"
            )

        # Create or update user
        user = create_or_update_oauth_user(
            db=db, provider=provider, oauth_id=oauth_id, email=email, name=name
        )

        # Create access token
        access_token = create_access_token({"sub": str(user.id)})

        return LoginResponse(access_token=access_token, user=user.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OAuth callback error", provider=provider, error=str(e))
        raise HTTPException(status_code=500, detail="OAuth authentication failed")


@router.post("/logout")
async def logout(response: Response, current_user: User = Depends(get_current_user)):
    """Logout user"""
    # In a real app, you might want to invalidate the token
    logger.info("User logged out", user_id=str(current_user.id))
    return {"message": "Logged out successfully"}


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current user info"""
    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        name=current_user.name,
        created_at=current_user.created_at.isoformat(),
    )


@router.put("/me", response_model=UserResponse)
async def update_me(
    name: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update current user info"""
    if name is not None:
        current_user.name = name
        db.commit()
        db.refresh(current_user)
        logger.info("User updated", user_id=str(current_user.id))

    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        name=current_user.name,
        created_at=current_user.created_at.isoformat(),
    )
