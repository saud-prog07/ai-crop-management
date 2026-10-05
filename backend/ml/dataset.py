from pathlib import Path
from typing import Tuple

from sklearn.model_selection import GroupShuffleSplit
from torchvision.datasets import ImageFolder

from ml.class_mapping import TOMATO_CLASSES


def discover_dataset(root: Path) -> ImageFolder:
    dataset = ImageFolder(root=str(root))
    found = tuple(dataset.classes)
    if found != TOMATO_CLASSES:
        raise ValueError(
            "Dataset classes do not exactly match the approved tomato class mapping. "
            f"Found: {found}"
        )
    return dataset


def stratified_indices(dataset: ImageFolder, validation_size: float = 0.2) -> Tuple[list[int], list[int]]:
    indices = list(range(len(dataset.samples)))
    labels = [label for _, label in dataset.samples]
    groups = [
        Path(path).name.split("___", maxsplit=1)[0]
        for path, _ in dataset.samples
    ]
    splitter = GroupShuffleSplit(n_splits=1, test_size=validation_size, random_state=42)
    train_indices, validation_indices = next(
        splitter.split(indices, labels, groups)
    )
    return list(train_indices), list(validation_indices)
