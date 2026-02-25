import os
import jwt
from typing import Dict, Optional
from datetime import datetime, timezone, timedelta


def _get_secret() -> str:
    secret = os.getenv("APP_SECRET", "")
    if secret and secret != "dev-secret-do-not-use-in-prod":
        return secret
    # Allow local dev / test usage with a fallback key
    if os.getenv("TESTING", "").lower() in ("1", "true", "yes"):
        return "test-secret-key-do-not-use-in-prod"
    import logging
    logging.getLogger(__name__).warning(
        "APP_SECRET is not set — using an insecure dev key. "
        "Set APP_SECRET to a secure value before deploying to production."
    )
    return "dev-secret-do-not-use-in-prod"


def create_token(payload: Dict[str, str], expires_minutes: int = 1440) -> str:
    """Create a JWT token with expiry (default 24h)."""
    secret = _get_secret()
    data = dict(payload)
    data["exp"] = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)
    data["iat"] = datetime.now(timezone.utc)
    return jwt.encode(data, secret, algorithm="HS256")


def verify_token(token: str) -> Optional[Dict[str, str]]:
    """Verify and decode a JWT token. Returns None on any failure."""
    try:
        secret = _get_secret()
        data = jwt.decode(token, secret, algorithms=["HS256"])
        return {k: str(v) for k, v in data.items() if k not in ("exp", "iat")}
    except Exception:
        return None
