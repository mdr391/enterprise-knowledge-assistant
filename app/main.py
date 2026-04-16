"""
Enterprise Knowledge Assistant — Main Application Entry Point
"""

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import query, ingest, health
from app.core.config import settings
from app.core.logging import get_logger
from app.core.vector_store import get_vector_store

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("Starting Enterprise Knowledge Assistant", version=settings.APP_VERSION)
    # Ensure vector store collection exists on boot
    store = get_vector_store()
    store.ensure_collection()
    logger.info("Vector store ready", collection=settings.CHROMA_COLLECTION)
    yield
    logger.info("Shutting down Enterprise Knowledge Assistant")


app = FastAPI(
    title="Enterprise Knowledge Assistant",
    description=(
        "A production-grade RAG service that lets employees query internal "
        "documents via a streaming LLM interface. Built to demonstrate "
        "responsible GenAI integration for enterprise workforce tooling."
    ),
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_and_timing(request: Request, call_next):
    """Attach a request ID and log latency for every request."""
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - start) * 1000, 1)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = str(elapsed)
    logger.info(
        "request completed",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        latency_ms=elapsed,
    )
    return response


# ── Exception handlers ────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("unhandled exception", path=request.url.path, error=str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": getattr(request.state, "request_id", "unknown")},
    )


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(ingest.router, prefix="/ingest", tags=["Ingestion"])
app.include_router(query.router, prefix="/query", tags=["Query"])
