from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ml.inference import TomatoClassifier


@dataclass(frozen=True)
class VisionInferenceResult:
    """Contract for the future OpenCV/YOLO/CNN inference implementation."""

    disease_or_pest: str
    confidence: float
    bounding_box: Optional[dict[str, float]]
    severity: Optional[str]


class DiseaseDetectionService:
    def __init__(self, checkpoint_path: Path) -> None:
        self._classifier = TomatoClassifier(checkpoint_path)

    def load(self) -> None:
        self._classifier.load()

    def infer(self, image_bytes: bytes, crop: str) -> VisionInferenceResult:
        if crop.strip().lower() != "tomato":
            raise ValueError("unsupported_crop: the first CNN currently supports tomato only.")
        prediction = self._classifier.predict(image_bytes)
        return VisionInferenceResult(
            disease_or_pest=prediction.disease,
            confidence=prediction.confidence,
            bounding_box=None,
            severity=None,
        )

    @property
    def available(self) -> bool:
        return self._classifier.available
