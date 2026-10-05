"""Train the approved PlantVillage tomato classifier.

Expected layout:
    data/train/<class-name>/*.jpg
    data/test/<class-name>/*.jpg
"""

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from ml.class_mapping import TOMATO_CLASSES
from ml.config import DEFAULT_MODEL_PATH
from ml.dataset import stratified_indices
from ml.model import build_model, load_checkpoint
from ml.preprocessing import evaluation_transforms, training_transforms


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += int((outputs.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)
    return total_loss / total, correct / total


def collect_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[list[int], list[int]]:
    actual: list[int] = []
    predicted: list[int] = []
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            outputs = model(images.to(device))
            actual.extend(labels.tolist())
            predicted.extend(outputs.argmax(dim=1).cpu().tolist())
    return actual, predicted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    raw_train = ImageFolder(str(args.train_root), transform=training_transforms())
    if tuple(raw_train.classes) != TOMATO_CLASSES:
        raise ValueError("Training dataset classes do not match the approved mapping.")
    eval_train = ImageFolder(str(args.train_root), transform=evaluation_transforms())
    train_indices, validation_indices = stratified_indices(eval_train)
    train_loader = DataLoader(
        Subset(raw_train, train_indices), batch_size=args.batch_size, shuffle=True
    )
    validation_loader = DataLoader(
        Subset(eval_train, validation_indices), batch_size=args.batch_size
    )
    test_dataset = ImageFolder(str(args.test_root), transform=evaluation_transforms())
    if tuple(test_dataset.classes) != TOMATO_CLASSES:
        raise ValueError("Test dataset classes do not match the approved mapping.")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_counts = {
        class_name: sum(1 for _, label in raw_train.samples if label == index)
        for index, class_name in enumerate(TOMATO_CLASSES)
    }
    validation_counts = {
        class_name: sum(1 for index in validation_indices if eval_train.samples[index][1] == label)
        for label, class_name in enumerate(TOMATO_CLASSES)
    }
    test_counts = {
        class_name: sum(1 for _, label in test_dataset.samples if label == index)
        for index, class_name in enumerate(TOMATO_CLASSES)
    }
    configuration = {
        "dataset_root": str(args.train_root.parent),
        "train_root": str(args.train_root),
        "test_root": str(args.test_root),
        "classes": list(TOMATO_CLASSES),
        "train_count": len(train_indices),
        "validation_count": len(validation_indices),
        "test_count": len(test_dataset),
        "train_class_counts": train_counts,
        "validation_class_counts": validation_counts,
        "test_class_counts": test_counts,
        "validation_method": (
            "GroupShuffleSplit(test_size=0.2, random_state=42), "
            "group=filename prefix before '___'"
        ),
        "image_size": 224,
        "batch_size": args.batch_size,
        "learning_rate": 1e-4,
        "optimizer": "Adam",
        "scheduler": None,
        "epochs": args.epochs,
        "transfer_learning": "ImageNet-pretrained ResNet18, full network fine-tuning",
        "device": str(device),
    }
    print(json.dumps({"training_configuration": configuration}, indent=2))

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    history: list[dict[str, float]] = []
    best_validation_accuracy = -1.0
    best_epoch = 0
    for epoch in range(args.epochs):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer, device)
        validation_loss, validation_accuracy = run_epoch(
            model, validation_loader, criterion, None, device
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        print(json.dumps(history[-1]))
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch + 1
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": TOMATO_CLASSES,
                    "image_size": 224,
                    "device": str(device),
                    "best_epoch": best_epoch,
                    "best_validation_accuracy": best_validation_accuracy,
                    "configuration": configuration,
                },
                args.output,
            )

    best_model = load_checkpoint(args.output, device)
    test_loss, test_accuracy = run_epoch(
        best_model, test_loader, criterion, None, device
    )
    validation_actual, validation_predicted = collect_predictions(
        best_model, validation_loader, device
    )
    test_actual, test_predicted = collect_predictions(best_model, test_loader, device)
    validation_report = classification_report(
        validation_actual,
        validation_predicted,
        target_names=list(TOMATO_CLASSES),
        output_dict=True,
        zero_division=0,
    )
    test_report = classification_report(
        test_actual,
        test_predicted,
        target_names=list(TOMATO_CLASSES),
        output_dict=True,
        zero_division=0,
    )
    report_path = args.output.parent.parent / "reports" / "tomato-training-evaluation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "history": history,
                "final": {
                    "best_epoch": best_epoch,
                    "best_validation_accuracy": best_validation_accuracy,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                },
                "configuration": configuration,
                "validation": {
                    "classification": validation_report,
                    "confusion_matrix": confusion_matrix(
                        validation_actual, validation_predicted
                    ).tolist(),
                },
                "test": {
                    "classification": test_report,
                    "confusion_matrix": confusion_matrix(
                        test_actual, test_predicted
                    ).tolist(),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"test_loss": test_loss, "test_accuracy": test_accuracy}))
    print(f"Saved evaluation report to {report_path}")
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
