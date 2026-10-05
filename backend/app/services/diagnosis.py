import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from bson import ObjectId
from fastapi import UploadFile

from app.db.mongo import get_database
from app.models.diagnosis import DiagnosisResponse, DiagnosisStatus
from app.services.vision import DiseaseDetectionService


class DiagnosisService:
    collection_name = "diagnoses"

    def __init__(self, vision_service: DiseaseDetectionService) -> None:
        self.vision_service = vision_service

    async def create_pending_diagnosis(
        self,
        image: UploadFile,
        crop: str,
        farmer_id: Optional[str],
        context_json: str,
    ) -> DiagnosisResponse:
        context = self._parse_context(context_json)
        image_bytes = await image.read()
        if not image_bytes:
            raise ValueError("The uploaded image is empty.")
        if not image.content_type or not image.content_type.startswith("image/"):
            raise ValueError("The uploaded file must be an image.")
        prediction = self.vision_service.infer(image_bytes, crop)

        now = datetime.now(timezone.utc)
        document: Dict[str, Any] = {
            "farmer_id": farmer_id,
            "crop": crop,
            "image": {
                "filename": image.filename or "uploaded-image",
                "content_type": image.content_type or "application/octet-stream",
                "size_bytes": len(image_bytes),
            },
            "context": context,
            "status": DiagnosisStatus.COMPLETED.value,
            "result": {
                "crop": crop,
                "disease_or_pest": prediction.disease_or_pest,
                "confidence": prediction.confidence,
                "bounding_box": prediction.bounding_box,
                "severity": prediction.severity,
            },
            "created_at": now,
            "updated_at": now,
        }
        result = get_database()[self.collection_name].insert_one(document)
        return self._to_response({**document, "_id": result.inserted_id})

    def get_diagnosis(self, diagnosis_id: str) -> DiagnosisResponse:
        if not ObjectId.is_valid(diagnosis_id):
            raise ValueError("The diagnosis ID is invalid.")
        document = get_database()[self.collection_name].find_one(
            {"_id": ObjectId(diagnosis_id)}
        )
        if document is None:
            raise LookupError("Diagnosis was not found.")
        return self._to_response(document)

    @staticmethod
    def _parse_context(context_json: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(context_json) if context_json else {}
        except json.JSONDecodeError as exc:
            raise ValueError("context_json must contain valid JSON.") from exc
        if not isinstance(parsed, dict):
            raise ValueError("context_json must contain a JSON object.")
        return parsed

    @staticmethod
    def _to_response(document: Dict[str, Any]) -> DiagnosisResponse:
        image = document["image"]
        return DiagnosisResponse(
            id=str(document["_id"]),
            farmer_id=document.get("farmer_id"),
            crop=document["crop"],
            image_filename=image["filename"],
            image_content_type=image["content_type"],
            image_size_bytes=image["size_bytes"],
            context=document.get("context", {}),
            status=DiagnosisStatus(document["status"]),
            result=document.get("result"),
            created_at=document["created_at"],
            updated_at=document["updated_at"],
        )
