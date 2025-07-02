# src/api/models/user.py
from sqlalchemy import Column, String, DateTime, func, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from src.api.dependencies.database import Base


class User(Base):
    """User model"""

    __tablename__ = "users"

    id = Column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False
    )
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255))
    oauth_provider = Column(String(50))
    oauth_id = Column(String(255))
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    last_login = Column(DateTime)

    # Relationships
    chat_sessions = relationship(
        "ChatSession", back_populates="user", cascade="all, delete-orphan"
    )
    datasets = relationship(
        "Dataset", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<User {self.email}>"

    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": str(self.id),
            "email": self.email,
            "name": self.name,
            "oauth_provider": self.oauth_provider,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }


class ChatSession(Base):
    """Chat session model"""

    __tablename__ = "chat_sessions"

    id = Column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False
    )
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title = Column(String(255))
    dataset_id = Column(UUID(as_uuid=True))
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )
    status = Column(String(50), default="active")

    # Relationships
    user = relationship("User", back_populates="chat_sessions")
    messages = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
    )
    experiments = relationship(
        "Experiment", back_populates="session", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<ChatSession {self.id}>"

    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "title": self.title,
            "dataset_id": str(self.dataset_id) if self.dataset_id else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "status": self.status,
            "message_count": len(self.messages) if self.messages else 0,
        }


class ChatMessage(Base):
    """Chat message model"""

    __tablename__ = "chat_messages"

    id = Column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False
    )
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role = Column(String(50), nullable=False)  # user, assistant, system, agent
    content = Column(String, nullable=False)
    metadata = Column(JSON)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    session = relationship("ChatSession", back_populates="messages")

    def __repr__(self):
        return f"<ChatMessage {self.id} - {self.role}>"

    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": str(self.id),
            "session_id": str(self.session_id),
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
