from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import get_settings
from app.core.errors import database_error_handler, runtime_error_handler
from app.db.mongo import close_mongodb, connect_to_mongodb
from app.routers import diagnoses, health
from pymongo.errors import PyMongoError
from app.routers.diagnoses import service as diagnosis_service


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    if settings.mongodb_uri:
        connect_to_mongodb(settings)
    try:
        diagnosis_service.vision_service.load()
    except RuntimeError:
        pass
    yield
    close_mongodb()


app = FastAPI(
    title=get_settings().app_name,
    version="0.1.0",
    lifespan=lifespan,
)

app.add_exception_handler(PyMongoError, database_error_handler)
app.add_exception_handler(RuntimeError, runtime_error_handler)
app.include_router(health.router, prefix="/api/v1")
app.include_router(diagnoses.router, prefix="/api/v1")
