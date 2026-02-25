from fastapi import APIRouter, HTTPException, Header
from typing import Optional
import os

from server.services.observability import observability


router = APIRouter(prefix="/internal", tags=["internal"])


def _require_admin_if_configured(x_admin_token: Optional[str]) -> None:
    expected = os.getenv("ADMIN_TOKEN")
    if expected and x_admin_token != expected:
        raise HTTPException(status_code=401, detail="unauthorized")


@router.get("/metrics", summary="Internal metrics", description="Returns in-memory counters, timer summaries, and recent orchestration traces.")
def get_metrics(x_admin_token: Optional[str] = Header(None)):
    _require_admin_if_configured(x_admin_token)
    return observability.snapshot()


@router.post("/metrics/reset", summary="Reset internal metrics", description="Clears in-memory counters and traces (admin-protected when ADMIN_TOKEN is set).")
def reset_metrics(x_admin_token: Optional[str] = Header(None)):
    _require_admin_if_configured(x_admin_token)
    observability.reset()
    return {"status": "ok"}
