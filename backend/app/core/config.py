from functools import lru_cache
from os import getenv
from pathlib import Path

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseModel):
    app_name: str = Field(default="Crop Management API")
    app_env: str = Field(default="development")
    mongodb_uri: str = Field(default="")
    mongodb_database: str = Field(default="crop_management")
    mongodb_server_selection_timeout_ms: int = Field(default=3000)
    cnn_model_path: Path = Field(
        default=Path(__file__).resolve().parents[2] / "ml" / "artifacts" / "tomato_resnet18.pt"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings(
        app_name=getenv("APP_NAME", "Crop Management API"),
        app_env=getenv("APP_ENV", "development"),
        mongodb_uri=getenv("MONGODB_URI", ""),
        mongodb_database=getenv("MONGODB_DATABASE", "crop_management"),
        mongodb_server_selection_timeout_ms=int(
            getenv("MONGODB_SERVER_SELECTION_TIMEOUT_MS", "3000")
        ),
        cnn_model_path=Path(
            getenv(
                "CNN_MODEL_PATH",
                str(
                    Path(__file__).resolve().parents[2]
                    / "ml"
                    / "artifacts"
                    / "tomato_resnet18.pt"
                ),
            )
        ),
    )
