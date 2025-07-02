# src/api/dependencies/database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
from src.core.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Create database engine
engine = create_engine(
    settings.database_url, pool_pre_ping=True, pool_size=10, max_overflow=20
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency to get database session.
    Ensures proper cleanup after request.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error("Database session error", error=str(e))
        db.rollback()
        raise
    finally:
        db.close()


def init_database():
    """Initialize database tables"""
    try:
        # Import all models to ensure they are registered
        from src.api.models import user, dataset, experiment  # noqa

        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        raise


def check_database_connection():
    """Check if database is accessible"""
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        logger.info("Database connection successful")
        return True
    except Exception as e:
        logger.error("Database connection failed", error=str(e))
        return False
