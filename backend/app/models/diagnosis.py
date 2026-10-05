from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class DiagnosisStatus(str, Enum):
    PENDING_MODEL_INFERENCE = "pending_model_inference"
    COMPLETED = "completed"
    FAILED = "failed"


class DiagnosisResult(BaseModel):
    crop: str
    disease_or_pest: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    bounding_box: Optional[Dict[str, float]] = None
    severity: Optional[str] = None


class DiagnosisResponse(BaseModel):
    id: str
    farmer_id: Optional[str] = None
    crop: str
    image_filename: str
    image_content_type: str
    image_size_bytes: int
    context: Dict[str, Any] = Field(default_factory=dict)
    status: DiagnosisStatus
    result: Optional[DiagnosisResult] = None
    created_at: datetime
    updated_at: datetime
