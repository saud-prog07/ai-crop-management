from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from ml.class_mapping import TOMATO_CLASSES
from ml.model import load_checkpoint
from ml.preprocessing import evaluation_transforms


class ModelUnavailableError(RuntimeError):
    """Raised when a trained CNN artifact is not available or valid."""


@dataclass(frozen=True)
class ClassificationPrediction:
    disease: str
    confidence: float


class TomatoClassifier:
    def __init__(self, checkpoint_path: Path) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._transform = evaluation_transforms()

    @property
    def available(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        try:
            self._model = load_checkpoint(self.checkpoint_path, self.device)
        except (FileNotFoundError, ValueError, OSError) as exc:
            self._model = None
            raise ModelUnavailableError(str(exc)) from exc

    def predict(self, image_bytes: bytes) -> ClassificationPrediction:
        if self._model is None:
            raise ModelUnavailableError(
                "CNN model is unavailable. Train the model before creating a diagnosis."
            )
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("The uploaded file is not a readable image.") from exc

        tensor = self._transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self._model(tensor), dim=1)[0]
        index = int(torch.argmax(probabilities).item())
        return ClassificationPrediction(
            disease=TOMATO_CLASSES[index],
            confidence=float(probabilities[index].item()),
        )
