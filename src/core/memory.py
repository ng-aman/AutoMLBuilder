# src/core/memory.py
import json
import pickle
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import redis
from abc import ABC, abstractmethod
from src.core.config import settings
from src.utils.logger import get_logger
from src.core.state import ConversationState

logger = get_logger(__name__)


class MemoryBackend(ABC):
    """Abstract base class for memory backends"""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key"""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value with optional TTL in seconds"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value by key"""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass

    @abstractmethod
    async def get_keys(self, pattern: str) -> List[str]:
        """Get keys matching pattern"""
        pass


class RedisMemoryBackend(MemoryBackend):
    """Redis-based memory backend"""

    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or settings.redis_url
        self.client = None
        self._connect()

    def _connect(self):
        """Connect to Redis"""
        try:
            self.client = redis.from_url(
                self.redis_url, decode_responses=False, socket_connect_timeout=5
            )
            self.client.ping()
            logger.info("Connected to Redis", url=self.redis_url)
        except Exception as e:
            logger.error("Failed to connect to Redis", error=str(e))
            raise

    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis"""
        try:
            value = self.client.get(key)
            if value:
                return pickle.loads(value)
            return None
        except Exception as e:
            logger.error("Redis get error", key=key, error=str(e))
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis"""
        try:
            serialized = pickle.dumps(value)
            if ttl:
                return self.client.setex(key, ttl, serialized)
            else:
                return self.client.set(key, serialized)
        except Exception as e:
            logger.error("Redis set error", key=key, error=str(e))
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from Redis"""
        try:
            return bool(self.client.delete(key))
        except Exception as e:
            logger.error("Redis delete error", key=key, error=str(e))
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            logger.error("Redis exists error", key=key, error=str(e))
            return False

    async def get_keys(self, pattern: str) -> List[str]:
        """Get keys matching pattern"""
        try:
            keys = self.client.keys(pattern)
            return [key.decode() if isinstance(key, bytes) else key for key in keys]
        except Exception as e:
            logger.error("Redis keys error", pattern=pattern, error=str(e))
            return []


class InMemoryBackend(MemoryBackend):
    """In-memory backend for development/testing"""

    def __init__(self):
        self.store: Dict[str, Any] = {}
        self.ttls: Dict[str, datetime] = {}

    def _cleanup_expired(self):
        """Remove expired keys"""
        now = datetime.utcnow()
        expired_keys = [key for key, expiry in self.ttls.items() if expiry <= now]
        for key in expired_keys:
            self.store.pop(key, None)
            self.ttls.pop(key, None)

    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory"""
        self._cleanup_expired()
        return self.store.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory"""
        self.store[key] = value
        if ttl:
            self.ttls[key] = datetime.utcnow() + timedelta(seconds=ttl)
        return True

    async def delete(self, key: str) -> bool:
        """Delete key from memory"""
        deleted = key in self.store
        self.store.pop(key, None)
        self.ttls.pop(key, None)
        return deleted

    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        self._cleanup_expired()
        return key in self.store

    async def get_keys(self, pattern: str) -> List[str]:
        """Get keys matching pattern"""
        self._cleanup_expired()
        # Simple pattern matching (supports * wildcard)
        import fnmatch

        return [key for key in self.store.keys() if fnmatch.fnmatch(key, pattern)]


class ConversationMemory:
    """High-level conversation memory manager"""

    def __init__(self, backend: Optional[MemoryBackend] = None):
        if backend:
            self.backend = backend
        elif settings.langraph_memory_type == "redis":
            self.backend = RedisMemoryBackend()
        else:
            self.backend = InMemoryBackend()

    def _get_state_key(self, session_id: str) -> str:
        """Get Redis key for state"""
        return f"state:{session_id}"

    def _get_checkpoint_key(self, session_id: str, checkpoint_id: str) -> str:
        """Get Redis key for checkpoint"""
        return f"checkpoint:{session_id}:{checkpoint_id}"

    def _get_lock_key(self, session_id: str) -> str:
        """Get Redis key for session lock"""
        return f"lock:{session_id}"

    async def save_state(self, session_id: str, state: ConversationState) -> bool:
        """Save conversation state"""
        try:
            key = self._get_state_key(session_id)
            # Keep state for 24 hours
            success = await self.backend.set(key, state, ttl=86400)
            if success:
                logger.debug("Saved conversation state", session_id=session_id)
            return success
        except Exception as e:
            logger.error("Failed to save state", session_id=session_id, error=str(e))
            return False

    async def load_state(self, session_id: str) -> Optional[ConversationState]:
        """Load conversation state"""
        try:
            key = self._get_state_key(session_id)
            state = await self.backend.get(key)
            if state:
                logger.debug("Loaded conversation state", session_id=session_id)
            return state
        except Exception as e:
            logger.error("Failed to load state", session_id=session_id, error=str(e))
            return None

    async def delete_state(self, session_id: str) -> bool:
        """Delete conversation state"""
        try:
            key = self._get_state_key(session_id)
            success = await self.backend.delete(key)
            if success:
                logger.debug("Deleted conversation state", session_id=session_id)
            return success
        except Exception as e:
            logger.error("Failed to delete state", session_id=session_id, error=str(e))
            return False

    async def create_checkpoint(
        self, session_id: str, checkpoint_id: str, state: ConversationState
    ) -> bool:
        """Create a checkpoint of the current state"""
        try:
            key = self._get_checkpoint_key(session_id, checkpoint_id)
            # Keep checkpoints for 7 days
            success = await self.backend.set(key, state, ttl=604800)
            if success:
                logger.info(
                    "Created checkpoint",
                    session_id=session_id,
                    checkpoint_id=checkpoint_id,
                )
            return success
        except Exception as e:
            logger.error(
                "Failed to create checkpoint",
                session_id=session_id,
                checkpoint_id=checkpoint_id,
                error=str(e),
            )
            return False

    async def load_checkpoint(
        self, session_id: str, checkpoint_id: str
    ) -> Optional[ConversationState]:
        """Load a checkpoint"""
        try:
            key = self._get_checkpoint_key(session_id, checkpoint_id)
            state = await self.backend.get(key)
            if state:
                logger.info(
                    "Loaded checkpoint",
                    session_id=session_id,
                    checkpoint_id=checkpoint_id,
                )
            return state
        except Exception as e:
            logger.error(
                "Failed to load checkpoint",
                session_id=session_id,
                checkpoint_id=checkpoint_id,
                error=str(e),
            )
            return None

    async def list_checkpoints(self, session_id: str) -> List[str]:
        """List all checkpoints for a session"""
        try:
            pattern = f"checkpoint:{session_id}:*"
            keys = await self.backend.get_keys(pattern)
            # Extract checkpoint IDs from keys
            checkpoint_ids = [
                key.split(":")[-1]
                for key in keys
                if key.startswith(f"checkpoint:{session_id}:")
            ]
            return checkpoint_ids
        except Exception as e:
            logger.error(
                "Failed to list checkpoints", session_id=session_id, error=str(e)
            )
            return []

    async def acquire_lock(self, session_id: str, user_id: str, ttl: int = 300) -> bool:
        """Acquire a lock on a session (prevent concurrent access)"""
        try:
            key = self._get_lock_key(session_id)
            # Check if lock exists
            if await self.backend.exists(key):
                current_lock = await self.backend.get(key)
                if current_lock and current_lock != user_id:
                    logger.warning(
                        "Session locked by another user",
                        session_id=session_id,
                        locked_by=current_lock,
                        requesting_user=user_id,
                    )
                    return False

            # Set or refresh lock
            success = await self.backend.set(key, user_id, ttl=ttl)
            if success:
                logger.debug(
                    "Acquired session lock", session_id=session_id, user_id=user_id
                )
            return success
        except Exception as e:
            logger.error(
                "Failed to acquire lock",
                session_id=session_id,
                user_id=user_id,
                error=str(e),
            )
            return False

    async def release_lock(self, session_id: str, user_id: str) -> bool:
        """Release a session lock"""
        try:
            key = self._get_lock_key(session_id)
            current_lock = await self.backend.get(key)

            # Only release if we own the lock
            if current_lock == user_id:
                success = await self.backend.delete(key)
                if success:
                    logger.debug(
                        "Released session lock", session_id=session_id, user_id=user_id
                    )
                return success
            else:
                logger.warning(
                    "Cannot release lock owned by another user",
                    session_id=session_id,
                    lock_owner=current_lock,
                    requesting_user=user_id,
                )
                return False
        except Exception as e:
            logger.error(
                "Failed to release lock",
                session_id=session_id,
                user_id=user_id,
                error=str(e),
            )
            return False

    async def get_active_sessions(self, user_id: str) -> List[str]:
        """Get all active sessions for a user"""
        try:
            # This would typically query the database
            # For now, we'll search Redis for state keys
            pattern = "state:*"
            keys = await self.backend.get_keys(pattern)

            active_sessions = []
            for key in keys:
                state = await self.backend.get(key)
                if state and state.get("user_id") == user_id:
                    session_id = key.split(":")[-1]
                    active_sessions.append(session_id)

            return active_sessions
        except Exception as e:
            logger.error("Failed to get active sessions", user_id=user_id, error=str(e))
            return []


# Global memory instance
memory = ConversationMemory()
