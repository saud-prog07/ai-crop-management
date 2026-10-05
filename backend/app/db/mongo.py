from typing import Optional

from pymongo import ASCENDING, MongoClient
from pymongo.database import Database

from app.core.config import Settings

_client: Optional[MongoClient] = None
_database: Optional[Database] = None


def connect_to_mongodb(settings: Settings) -> None:
    global _client, _database
    _client = MongoClient(
        settings.mongodb_uri,
        serverSelectionTimeoutMS=settings.mongodb_server_selection_timeout_ms,
    )
    _client.admin.command("ping")
    _database = _client[settings.mongodb_database]
    _database.diagnoses.create_index([("created_at", ASCENDING)])


def close_mongodb() -> None:
    global _client, _database
    if _client is not None:
        _client.close()
    _client = None
    _database = None


def get_database() -> Database:
    if _database is None:
        raise RuntimeError(
            "MongoDB is not configured or the connection has not been established."
        )
    return _database


def is_database_connected() -> bool:
    if _client is None:
        return False
    try:
        _client.admin.command("ping")
    except Exception:
        return False
    return True
