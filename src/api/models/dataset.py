# src/api/models/dataset.py
from sqlalchemy import (
    Column,
    String,
    Integer,
    BigInteger,
    DateTime,
    ForeignKey,
    JSON,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid
from src.api.dependencies.database import Base


class Dataset(Base):
    """Dataset model"""

    __tablename__ = "datasets"

    id = Column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False
    )
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(BigInteger)
    rows_count = Column(Integer)
    columns_count = Column(Integer)
    metadata = Column(JSON)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    user = relationship("User", back_populates="datasets")

    def __repr__(self):
        return f"<Dataset {self.filename}>"

    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "filename": self.filename,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "rows_count": self.rows_count,
            "columns_count": self.columns_count,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
