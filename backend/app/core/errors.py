from fastapi import Request
from fastapi.responses import JSONResponse
from pymongo.errors import PyMongoError


async def database_error_handler(_: Request, exc: PyMongoError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "code": "database_unavailable",
                "message": "The database operation could not be completed.",
            }
        },
    )


async def runtime_error_handler(_: Request, exc: RuntimeError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "code": "service_unavailable",
                "message": str(exc),
            }
        },
    )
