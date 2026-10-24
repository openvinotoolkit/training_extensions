# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Download a ~30-sample CIFAR-10 subset and export it as a Datumaro experimental dataset.

The script:
1. Downloads CIFAR-10 via ``torchvision.datasets``.
2. Selects a balanced subset from 5 classes (6 samples per class) - enough for
   quick classification testing / benchmarking.
3. Wraps every image in a ``ClassificationSample`` (the library entity defined
   in ``getitune.data.entity.sample``).
4. Stores all samples in a ``datumaro.experimental.Dataset``.
5. Exports the dataset to disk with ``export_dataset``.

Usage
-----
    python tests/assets/classification_cifar10/download.py [--output_dir OUTPUT_DIR]

The default output directory is
``tests/assets/classification_cifar10``.
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datumaro.experimental import Dataset
from datumaro.experimental.categories import LabelCategories
from datumaro.experimental.export_import import export_dataset
from datumaro.experimental.fields import ImageInfo as DmImageInfo
from datumaro.experimental.fields import Subset
from torchvision import tv_tensors
from torchvision.datasets import CIFAR10

from getitune.data.entity.sample import ClassificationSample

# CIFAR-10 class names - use a 5-class subset to keep the dataset small
CIFAR10_CLASSES: tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# Indices of the classes we actually keep (5 of 10)
SELECTED_CLASS_INDICES: tuple[int, ...] = (0, 2, 4, 6, 8)  # airplane, bird, deer, frog, ship

SAMPLES_PER_CLASS_TRAIN: int = 4
SAMPLES_PER_CLASS_VAL: int = 1
SAMPLES_PER_CLASS_TEST: int = 1
# Total: 5 classes x (4 train + 1 val + 1 test) = 30 samples


def _select_samples(
    cifar_dataset: CIFAR10,
    samples_per_class: int,
) -> list[tuple[np.ndarray, int, int]]:
    """Select a balanced subset from a CIFAR-10 split (only selected classes).

    Returns a list of (image_array_HWC, remapped_label_idx, global_idx) tuples.
    The label index is remapped to a contiguous 0..N-1 range.
    """
    # Map original class index → contiguous index for the selected subset
    orig_to_new = {orig: new for new, orig in enumerate(SELECTED_CLASS_INDICES)}
    class_counts: dict[int, int] = defaultdict(int)
    selected: list[tuple[np.ndarray, int, int]] = []
    num_classes = len(SELECTED_CLASS_INDICES)

    for idx in range(len(cifar_dataset)):
        img, label = cifar_dataset[idx]
        if label not in orig_to_new:
            continue
        new_label = orig_to_new[label]
        if class_counts[new_label] >= samples_per_class:
            continue
        img_np: np.ndarray = np.array(img, dtype=np.uint8)
        selected.append((img_np, new_label, idx))
        class_counts[new_label] += 1
        if all(c >= samples_per_class for c in class_counts.values()) and len(class_counts) == num_classes:
            break

    return selected


def _build_dataset(
    train_items: list[tuple[np.ndarray, int, int]],
    val_items: list[tuple[np.ndarray, int, int]],
    test_items: list[tuple[np.ndarray, int, int]],
    images_dir: Path,
) -> Dataset:
    """Build a ``datumaro.experimental.Dataset`` of ``ClassificationSample`` objects."""
    from PIL import Image as PILImage

    selected_class_names = tuple(CIFAR10_CLASSES[i] for i in SELECTED_CLASS_INDICES)
    categories = {"label": LabelCategories(labels=selected_class_names)}
    dataset: Dataset = Dataset(ClassificationSample, categories=categories)  # type: ignore[arg-type]

    images_dir.mkdir(parents=True, exist_ok=True)

    def _add_items(
        items: list[tuple[np.ndarray, int, int]],
        subset: Subset,
        prefix: str,
    ) -> None:
        for img_np, label_idx, global_idx in items:
            h, w = img_np.shape[:2]
            # Convert HWC numpy to CHW torch tensor for tv_tensors.Image
            img_chw = torch.from_numpy(img_np.transpose(2, 0, 1))
            image = tv_tensors.Image(img_chw)

            # Save image to disk for reference / reproducibility
            filename = f"{prefix}_{global_idx:05d}.png"
            PILImage.fromarray(img_np).save(images_dir / filename)

            sample = ClassificationSample(
                image=image,
                label=torch.tensor(label_idx, dtype=torch.uint16),
                dm_image_info=DmImageInfo(width=w, height=h),
                subset=subset,
            )
            dataset.append(sample)

    _add_items(train_items, Subset.TRAINING, "train")
    _add_items(val_items, Subset.VALIDATION, "val")
    _add_items(test_items, Subset.TESTING, "test")

    return dataset


def main(output_dir: Path | None = None) -> None:
    """Download CIFAR-10, build a 30-sample classification dataset, and export it."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent

    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir = output_dir / "raw"
    images_dir = output_dir / "images"

    print(f"Downloading CIFAR-10 to {download_dir} ...")
    train_set = CIFAR10(root=str(download_dir), train=True, download=True)
    test_set = CIFAR10(root=str(download_dir), train=False, download=True)

    print("Selecting balanced subset ...")
    train_items = _select_samples(train_set, SAMPLES_PER_CLASS_TRAIN)
    valtest_items = _select_samples(test_set, SAMPLES_PER_CLASS_VAL + SAMPLES_PER_CLASS_TEST)
    num_selected = len(SELECTED_CLASS_INDICES)
    val_items = valtest_items[: SAMPLES_PER_CLASS_VAL * num_selected]
    test_items = valtest_items[SAMPLES_PER_CLASS_VAL * num_selected :]
    print(f"  Training   samples: {len(train_items)}")
    print(f"  Validation samples: {len(val_items)}")
    print(f"  Testing    samples: {len(test_items)}")
    print(f"  Total:              {len(train_items) + len(val_items) + len(test_items)}")

    print("Building datumaro experimental dataset with ClassificationSample ...")
    dataset = _build_dataset(train_items, val_items, test_items, images_dir)
    print(f"  Dataset length: {len(dataset)}")

    print(f"Exporting dataset to {output_dir} ...")
    export_dataset(dataset, output_dir)

    print("Cleaning up intermediate files ...")
    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(images_dir, ignore_errors=True)

    print("Done ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a 30-sample CIFAR-10 classification benchmark dataset.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save the dataset. Defaults to tests/assets/classification_cifar10",
    )
    args = parser.parse_args()
    main(output_dir=args.output_dir)
