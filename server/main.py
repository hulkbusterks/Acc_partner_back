from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from server.auth import router as auth_router
from server.utils.env import ensure_env_loaded
from server.routes.sessions import router as sessions_router
from server.routes.ingest import router as ingest_router
from server.routes.leaderboard import router as leaderboard_router
from server.routes.observability import router as observability_router
from server.db import create_db_and_tables, engine
from contextlib import asynccontextmanager
import os
import logging
from server.services.faiss_store import store as faiss_store

logger = logging.getLogger("server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_env_loaded()
    create_db_and_tables()
    if os.getenv("EMBEDDING_WARMUP", "false").strip().lower() in ("1", "true", "yes", "on"):
        faiss_store.warmup()
    yield


app = FastAPI(
    title="Accountability PoC",
    description="PoC backend for timed sessions, RAG MCQ prompts, and a simple leaderboard. See `/docs` for OpenAPI UI.",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS
allowed_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in allowed_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


app.include_router(auth_router)
app.include_router(sessions_router)
app.include_router(ingest_router)
app.include_router(leaderboard_router)
app.include_router(observability_router)


@app.get("/health")
def health_check():
    checks: dict[str, object] = {"status": "ok"}
    # DB connectivity check
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["db"] = "ok"
    except Exception as e:
        checks["db"] = f"error: {e}"
        checks["status"] = "degraded"
    # FAISS store check
    checks["faiss_vectors"] = faiss_store.index.ntotal if faiss_store.index else 0
    checks["faiss_items"] = len(faiss_store.items)
    return checks
