# src/api/routers/agents.py
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any
import uuid
from datetime import datetime
from pydantic import BaseModel
from src.api.dependencies.database import get_db
from src.api.dependencies.auth import get_current_user
from src.api.models.user import User, ChatSession, ChatMessage
from src.core.config import settings
from src.core.state import (
    create_initial_state,
    WorkflowMode,
    WorkflowStatus,
    add_message,
    resolve_approval,
)
from src.core.memory import memory
from src.core.events import event_tracker, EventType
from src.workflows.graph_builder import create_workflow_graph
from src.utils.logger import get_logger
from src.utils.exceptions import ResourceNotFoundError, ValidationError

logger = get_logger(__name__)

router = APIRouter(prefix="/api")


# Pydantic models
class ChatMessageRequest(BaseModel):
    """Chat message request"""

    message: str
    session_id: Optional[str] = None
    mode: WorkflowMode = WorkflowMode.AUTO
    dataset_id: Optional[str] = None


class ChatMessageResponse(BaseModel):
    """Chat message response"""

    session_id: str
    message: str
    metadata: Optional[Dict[str, Any]] = None
    requires_approval: bool = False
    approval_request: Optional[Dict[str, Any]] = None


class ApprovalRequest(BaseModel):
    """Approval request"""

    session_id: str
    decision: str
    reason: Optional[str] = None


class SessionListResponse(BaseModel):
    """Session list response"""

    sessions: List[Dict[str, Any]]
    total: int


class SessionDetailResponse(BaseModel):
    """Session detail response"""

    session: Dict[str, Any]
    messages: List[Dict[str, Any]]
    state: Optional[Dict[str, Any]] = None


# Global workflow graph instance
workflow_graph = None


def get_workflow_graph():
    """Get or create workflow graph"""
    global workflow_graph
    if workflow_graph is None:
        workflow_graph = create_workflow_graph()
    return workflow_graph


@router.post("/chat/message", response_model=ChatMessageResponse)
async def send_chat_message(
    request: ChatMessageRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Send a chat message and get response"""
    try:
        # Get or create session
        if request.session_id:
            session = (
                db.query(ChatSession)
                .filter(
                    ChatSession.id == request.session_id,
                    ChatSession.user_id == current_user.id,
                )
                .first()
            )
            if not session:
                raise ResourceNotFoundError("ChatSession", request.session_id)
        else:
            # Create new session
            session = ChatSession(
                user_id=current_user.id,
                title=(
                    request.message[:50] + "..."
                    if len(request.message) > 50
                    else request.message
                ),
                dataset_id=request.dataset_id,
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            logger.info("Created new chat session", session_id=str(session.id))

        # Store user message
        user_msg = ChatMessage(
            session_id=session.id, role="user", content=request.message
        )
        db.add(user_msg)
        db.commit()

        # Load or create state
        state = await memory.load_state(str(session.id))
        if not state:
            state = create_initial_state(
                session_id=str(session.id),
                user_id=str(current_user.id),
                mode=request.mode,
                debug_enabled=settings.enable_debug_mode,
            )
            if request.dataset_id:
                state["dataset_id"] = request.dataset_id

        # Add message to state
        state = add_message(state, "user", request.message)

        # Acquire lock
        if not await memory.acquire_lock(str(session.id), str(current_user.id)):
            raise HTTPException(
                status_code=409, detail="Session is locked by another process"
            )

        try:
            # Run workflow
            graph = get_workflow_graph()

            # Track workflow start
            await event_tracker.track_workflow_update(
                session_id=str(session.id),
                status="running",
                user_id=str(current_user.id),
            )

            # Execute workflow
            final_state = await graph.ainvoke(
                state, config={"recursion_limit": settings.langraph_max_iterations}
            )

            # Save state
            await memory.save_state(str(session.id), final_state)

            # Get response
            response_message = (
                final_state["messages"][-1]["content"]
                if final_state["messages"]
                else "I'm processing your request..."
            )

            # Check if approval is needed
            requires_approval = final_state.get("pending_approval") is not None
            approval_request = final_state.get("pending_approval")

            # Store assistant message
            assistant_msg = ChatMessage(
                session_id=session.id,
                role="assistant",
                content=response_message,
                metadata={
                    "requires_approval": requires_approval,
                    "approval_request": approval_request,
                },
            )
            db.add(assistant_msg)

            # Update session
            session.updated_at = datetime.utcnow()
            db.commit()

            return ChatMessageResponse(
                session_id=str(session.id),
                message=response_message,
                metadata={
                    "status": final_state["status"],
                    "models_trained": len(final_state.get("models_trained", [])),
                    "best_model": final_state.get("best_model"),
                },
                requires_approval=requires_approval,
                approval_request=approval_request,
            )

        finally:
            # Release lock
            await memory.release_lock(str(session.id), str(current_user.id))

    except Exception as e:
        logger.error("Chat message error", error=str(e), user_id=str(current_user.id))

        # Store error message
        if "session" in locals():
            error_msg = ChatMessage(
                session_id=session.id,
                role="system",
                content=f"Error: {str(e)}",
                metadata={"error": True},
            )
            db.add(error_msg)
            db.commit()

        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/approve")
async def approve_action(
    request: ApprovalRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Approve or reject a pending action"""
    try:
        # Verify session ownership
        session = (
            db.query(ChatSession)
            .filter(
                ChatSession.id == request.session_id,
                ChatSession.user_id == current_user.id,
            )
            .first()
        )
        if not session:
            raise ResourceNotFoundError("ChatSession", request.session_id)

        # Load state
        state = await memory.load_state(request.session_id)
        if not state:
            raise ValidationError("No active state for session")

        if not state.get("pending_approval"):
            raise ValidationError("No pending approval request")

        # Resolve approval
        state = resolve_approval(state, request.decision, request.reason)

        # Save state
        await memory.save_state(request.session_id, state)

        # Track approval
        await event_tracker.store.add_event(
            Event(
                event_type=EventType.APPROVAL_RESPONSE,
                session_id=request.session_id,
                data={"decision": request.decision, "reason": request.reason},
                user_id=str(current_user.id),
            )
        )

        # Continue workflow
        return await send_chat_message(
            ChatMessageRequest(
                message=f"[Approved: {request.decision}]", session_id=request.session_id
            ),
            current_user,
            db,
        )

    except Exception as e:
        logger.error("Approval error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/sessions", response_model=SessionListResponse)
async def get_chat_sessions(
    skip: int = 0,
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get user's chat sessions"""
    # Get total count
    total = db.query(ChatSession).filter(ChatSession.user_id == current_user.id).count()

    # Get sessions
    sessions = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == current_user.id)
        .order_by(ChatSession.updated_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    return SessionListResponse(sessions=[s.to_dict() for s in sessions], total=total)


@router.get("/chat/session/{session_id}", response_model=SessionDetailResponse)
async def get_session_detail(
    session_id: str,
    include_state: bool = False,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get session details with messages"""
    # Get session
    session = (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id)
        .first()
    )

    if not session:
        raise ResourceNotFoundError("ChatSession", session_id)

    # Get messages
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
        .all()
    )

    response = SessionDetailResponse(
        session=session.to_dict(), messages=[m.to_dict() for m in messages]
    )

    # Include state if requested
    if include_state:
        state = await memory.load_state(session_id)
        if state:
            # Remove sensitive information
            safe_state = {
                k: v for k, v in state.items() if k not in ["user_id", "agent_thoughts"]
            }
            response.state = safe_state

    return response


@router.delete("/chat/session/{session_id}")
async def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a chat session"""
    # Get session
    session = (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id)
        .first()
    )

    if not session:
        raise ResourceNotFoundError("ChatSession", session_id)

    # Delete from database
    db.delete(session)
    db.commit()

    # Delete from memory
    await memory.delete_state(session_id)
    await event_tracker.store.clear_events(session_id)

    logger.info("Deleted chat session", session_id=session_id)

    return {"message": "Session deleted successfully"}


@router.get("/debug/events/{session_id}")
async def get_debug_events(
    session_id: str,
    event_types: Optional[List[str]] = None,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get debug events for a session"""
    # Verify session ownership
    session = (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id)
        .first()
    )

    if not session:
        raise ResourceNotFoundError("ChatSession", session_id)

    # Get events
    events = await event_tracker.store.get_events(
        session_id=session_id,
        event_types=[EventType(et) for et in event_types] if event_types else None,
        limit=limit,
    )

    return {"session_id": session_id, "events": events, "count": len(events)}
