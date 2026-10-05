from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from ml.class_mapping import TOMATO_CLASSES


def build_model(pretrained: bool = True) -> nn.Module:
    model = resnet18(
        weights=ResNet18_Weights.DEFAULT if pretrained else None
    )
    model.fc = nn.Linear(model.fc.in_features, len(TOMATO_CLASSES))
    return model


def load_checkpoint(path: Path, device: torch.device) -> nn.Module:
    if not path.is_file():
        raise FileNotFoundError(f"CNN model checkpoint was not found: {path}")

    checkpoint: Any = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("CNN checkpoint is invalid: expected a state_dict.")
    if tuple(checkpoint.get("classes", ())) != TOMATO_CLASSES:
        raise ValueError("CNN checkpoint class mapping does not match the approved tomato classes.")

    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model
