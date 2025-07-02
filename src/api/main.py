# src/api/main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
from src.core.config import settings
from src.api.dependencies.database import init_database, check_database_connection
from src.utils.logger import get_logger
from src.utils.exceptions import APIException
from src.core.memory import memory
from src.core.events import event_tracker

# Import routers
from src.api.routers import auth, agents, datasets, experiments

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    # Startup
    logger.info("Starting AutoML Builder API...")

    # Initialize database
    if not check_database_connection():
        logger.error("Failed to connect to database")
        raise RuntimeError("Database connection failed")

    init_database()

    # Initialize MLflow
    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    logger.info(f"MLflow tracking URI: {settings.mlflow_tracking_uri}")

    # Test Redis connection
    try:
        await memory.backend.set("test_key", "test_value", ttl=10)
        test_value = await memory.backend.get("test_key")
        if test_value == "test_value":
            logger.info("Redis connection successful")
        await memory.backend.delete("test_key")
    except Exception as e:
        logger.error("Redis connection failed", error=str(e))
        if settings.is_production:
            raise RuntimeError("Redis connection failed")

    logger.info("API startup complete")

    yield

    # Shutdown
    logger.info("Shutting down AutoML Builder API...")


# Create FastAPI app
app = FastAPI(
    title="AutoML Builder API",
    description="AI-powered AutoML platform with multi-agent architecture",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.enable_debug_mode else None,
    redoc_url="/redoc" if settings.enable_debug_mode else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)

    # Log request
    logger.info(
        "API Request",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        process_time=process_time,
    )

    return response


# Global exception handler
@app.exception_handler(APIException)
async def api_exception_handler(request: Request, exc: APIException):
    """Handle API exceptions"""
    return JSONResponse(status_code=exc.status_code, content=exc.detail)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "environment": settings.api_environment,
        "database": "connected" if check_database_connection() else "disconnected",
        "features": {
            "debug_mode": settings.enable_debug_mode,
            "auto_mode": settings.enable_auto_mode,
            "max_upload_size_mb": settings.max_upload_size_mb,
        },
    }


# Include routers
app.include_router(auth.router, tags=["Authentication"])
app.include_router(agents.router, tags=["Agents"])
app.include_router(datasets.router, tags=["Datasets"])
app.include_router(experiments.router, tags=["Experiments"])


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to AutoML Builder API",
        "version": "1.0.0",
        "docs": "/docs" if settings.enable_debug_mode else None,
    }


# WebSocket endpoint for debug events
@app.websocket("/ws/debug/{session_id}")
async def debug_websocket(websocket, session_id: str):
    """WebSocket endpoint for real-time debug events"""
    from fastapi import WebSocket, WebSocketDisconnect

    await websocket.accept()
    logger.info("Debug WebSocket connected", session_id=session_id)

    # Subscribe to events
    async def send_event(event):
        try:
            await websocket.send_json(event.to_dict())
        except:
            pass

    event_tracker.store.subscribe(session_id, send_event)

    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info("Debug WebSocket disconnected", session_id=session_id)
    finally:
        event_tracker.store.unsubscribe(session_id, send_event)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.is_development,
        log_level=settings.log_level.lower(),
    )
