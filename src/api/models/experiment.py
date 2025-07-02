# src/api/models/experiment.py
from sqlalchemy import Column, String, DateTime, ForeignKey, JSON, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from src.api.dependencies.database import Base


class Experiment(Base):
    """ML experiment model"""

    __tablename__ = "experiments"

    id = Column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False
    )
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    mlflow_run_id = Column(String(255), unique=True)
    status = Column(String(50))  # running, completed, failed
    results = Column(JSON)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    completed_at = Column(DateTime)

    # Relationships
    session = relationship("ChatSession", back_populates="experiments")

    def __repr__(self):
        return f"<Experiment {self.id}>"

    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": str(self.id),
            "session_id": str(self.session_id),
            "mlflow_run_id": self.mlflow_run_id,
            "status": self.status,
            "results": self.results,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
        }
