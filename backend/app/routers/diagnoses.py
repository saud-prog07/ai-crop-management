from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.models.diagnosis import DiagnosisResponse
from app.services.diagnosis import DiagnosisService
from app.services.vision import DiseaseDetectionService
from app.core.config import get_settings

router = APIRouter(prefix="/diagnoses", tags=["diagnoses"])
service = DiagnosisService(DiseaseDetectionService(get_settings().cnn_model_path))


@router.post(
    "",
    response_model=DiagnosisResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_diagnosis(
    image: UploadFile = File(...),
    crop: str = Form(..., min_length=1),
    farmer_id: Optional[str] = Form(default=None),
    context_json: str = Form(default="{}"),
) -> DiagnosisResponse:
    try:
        return await service.create_pending_diagnosis(
            image=image,
            crop=crop,
            farmer_id=farmer_id,
            context_json=context_json,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/{diagnosis_id}", response_model=DiagnosisResponse)
def get_diagnosis(diagnosis_id: str) -> DiagnosisResponse:
    try:
        return service.get_diagnosis(diagnosis_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
