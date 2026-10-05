from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = BACKEND_ROOT / "ml" / "artifacts" / "tomato_resnet18.pt"
IMAGE_SIZE = 224
