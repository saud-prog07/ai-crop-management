from fastapi import APIRouter

from app.core.config import get_settings
from app.db.mongo import is_database_connected

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    database_configured = bool(settings.mongodb_uri)
    database_connected = is_database_connected()
    return {
        "status": "ok" if database_connected else "degraded",
        "database": (
            "connected"
            if database_connected
            else "not_configured" if not database_configured else "unavailable"
        ),
    }
