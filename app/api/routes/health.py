"""Health check endpoints — liveness and readiness probes."""

import time
from fastapi import APIRouter
from app.core.config import settings
from app.core.models import HealthResponse, HealthStatus, ComponentHealth
from app.core.vector_store import get_vector_store
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

_START_TIME = time.time()


@router.get("/", response_model=HealthResponse, summary="Full health check")
async def health():
    components: dict[str, ComponentHealth] = {}

    # Vector store check
    try:
        store = get_vector_store()
        count = store.count()
        components["vector_store"] = ComponentHealth(
            status=HealthStatus.OK,
            detail=f"{count} chunks indexed",
        )
    except Exception as e:
        components["vector_store"] = ComponentHealth(status=HealthStatus.DOWN, detail=str(e))

    # LLM config check (just verify key is set — don't call the API)
    if settings.ANTHROPIC_API_KEY:
        components["llm"] = ComponentHealth(status=HealthStatus.OK, detail=settings.LLM_MODEL)
    else:
        components["llm"] = ComponentHealth(status=HealthStatus.DEGRADED, detail="ANTHROPIC_API_KEY not set")

    if settings.OPENAI_API_KEY:
        components["embeddings"] = ComponentHealth(status=HealthStatus.OK, detail=settings.EMBEDDING_MODEL)
    else:
        components["embeddings"] = ComponentHealth(status=HealthStatus.DEGRADED, detail="OPENAI_API_KEY not set")

    overall = HealthStatus.OK
    if any(c.status == HealthStatus.DOWN for c in components.values()):
        overall = HealthStatus.DOWN
    elif any(c.status == HealthStatus.DEGRADED for c in components.values()):
        overall = HealthStatus.DEGRADED

    return HealthResponse(
        status=overall,
        version=settings.APP_VERSION,
        components=components,
        uptime_seconds=round(time.time() - _START_TIME, 1),
    )


@router.get("/live", summary="Liveness probe (Kubernetes)")
async def liveness():
    return {"status": "alive"}


@router.get("/ready", summary="Readiness probe (Kubernetes)")
async def readiness():
    try:
        store = get_vector_store()
        store.count()
        return {"status": "ready"}
    except Exception as e:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=503, content={"status": "not ready", "detail": str(e)})
