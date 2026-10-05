"""Evaluate a trained checkpoint and write classification metrics."""

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from ml.class_mapping import TOMATO_CLASSES
from ml.model import load_checkpoint
from ml.preprocessing import evaluation_transforms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("ml/reports/tomato-evaluation.json"))
    args = parser.parse_args()

    dataset = ImageFolder(str(args.test_root), transform=evaluation_transforms())
    if tuple(dataset.classes) != TOMATO_CLASSES:
        raise ValueError("Test dataset classes do not match the approved mapping.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint, device)
    loader = DataLoader(dataset, batch_size=32)
    actual: list[int] = []
    predicted: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            output = model(images.to(device))
            actual.extend(labels.tolist())
            predicted.extend(output.argmax(dim=1).cpu().tolist())

    report = classification_report(
        actual, predicted, target_names=list(TOMATO_CLASSES), output_dict=True, zero_division=0
    )
    result = {
        "overall_accuracy": report["accuracy"],
        "per_class": {name: report[name] for name in TOMATO_CLASSES},
        "confusion_matrix": confusion_matrix(actual, predicted).tolist(),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
