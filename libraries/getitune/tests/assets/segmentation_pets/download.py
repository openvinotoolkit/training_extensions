# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Download a ~50-sample Oxford-IIIT Pet subset for semantic segmentation.

The Oxford-IIIT Pet dataset includes pixel-level segmentation masks (trimap)
with three classes: foreground (pet), background, and boundary. This makes it
a lightweight, real-world dataset for testing ``SegmentationSample``.

Usage
-----
    python tests/assets/segmentation_pets/download.py [--output_dir OUTPUT_DIR]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from datumaro.experimental import Dataset
from datumaro.experimental.categories import Colormap, MaskCategories, RgbColor
from datumaro.experimental.export_import import export_dataset
from datumaro.experimental.fields import ImageInfo as DmImageInfo
from datumaro.experimental.fields import Subset
from PIL import Image as PILImage
from torchvision import tv_tensors
from torchvision.datasets import OxfordIIITPet

from getitune.data.entity.sample import SegmentationSample

# Oxford-IIIT Pet trimap classes
# Original trimap: 1 = foreground, 2 = background, 3 = boundary
# We remap to 0-based: 0 = background, 1 = pet, 2 = boundary
PET_SEG_CLASSES: tuple[str, ...] = ("background", "pet", "boundary")

TOTAL_TRAIN: int = 35
TOTAL_VAL: int = 8
TOTAL_TEST: int = 7

# Resize images to keep the exported dataset under 3 MB
IMG_SIZE: int = 64


def _select_samples(
    pet_dataset: OxfordIIITPet,
    total: int,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Select samples with segmentation masks.

    Returns a list of (image_HWC, mask_HW, global_idx) tuples.
    """
    selected: list[tuple[np.ndarray, np.ndarray, int]] = []

    for idx in range(len(pet_dataset)):
        if len(selected) >= total:
            break
        img, mask = pet_dataset[idx]
        img_np = np.array(img, dtype=np.uint8)
        mask_np = np.array(mask, dtype=np.uint8)

        # Resize image and mask to IMG_SIZE x IMG_SIZE
        img_pil = PILImage.fromarray(img_np).resize((IMG_SIZE, IMG_SIZE), PILImage.BILINEAR)  # type: ignore[attr-defined]
        mask_pil = PILImage.fromarray(mask_np).resize((IMG_SIZE, IMG_SIZE), PILImage.NEAREST)  # type: ignore[attr-defined]
        img_np = np.array(img_pil, dtype=np.uint8)
        mask_np = np.array(mask_pil, dtype=np.uint8)

        # Remap trimap: 1 -> 1 (pet/foreground), 2 -> 0 (background), 3 -> 2 (boundary)
        remapped = np.zeros_like(mask_np)
        remapped[mask_np == 1] = 1  # pet
        remapped[mask_np == 2] = 0  # background
        remapped[mask_np == 3] = 2  # boundary

        selected.append((img_np, remapped, idx))

    return selected


def _build_dataset(
    train_items: list[tuple[np.ndarray, np.ndarray, int]],
    val_items: list[tuple[np.ndarray, np.ndarray, int]],
    test_items: list[tuple[np.ndarray, np.ndarray, int]],
    images_dir: Path,
) -> Dataset:
    """Build a ``datumaro.experimental.Dataset`` of ``SegmentationSample``."""
    colormap = Colormap(
        {
            0: RgbColor(0, 0, 0),  # background
            1: RgbColor(128, 0, 0),  # pet
            2: RgbColor(0, 128, 0),  # boundary
        }
    )
    categories = {"masks": MaskCategories(labels=list(PET_SEG_CLASSES), colormap=colormap)}
    dataset: Dataset = Dataset(SegmentationSample, categories=categories)  # type: ignore[arg-type]

    images_dir.mkdir(parents=True, exist_ok=True)

    def _add_items(
        items: list[tuple[np.ndarray, np.ndarray, int]],
        subset: Subset,
        prefix: str,
    ) -> None:
        for img_np, mask_np, global_idx in items:
            h, w = img_np.shape[:2]
            image = tv_tensors.Image(torch.from_numpy(img_np.transpose(2, 0, 1)))

            # Mask shape: (1, H, W) -- single-channel class index mask
            masks = tv_tensors.Mask(torch.from_numpy(mask_np[np.newaxis, :, :]))

            filename = f"{prefix}_{global_idx:05d}.png"
            PILImage.fromarray(img_np).save(images_dir / filename)

            sample = SegmentationSample(
                image=image,
                masks=masks,
                dm_image_info=DmImageInfo(width=w, height=h),
                subset=subset,
            )
            dataset.append(sample)

    _add_items(train_items, Subset.TRAINING, "train")
    _add_items(val_items, Subset.VALIDATION, "val")
    _add_items(test_items, Subset.TESTING, "test")

    return dataset


def main(output_dir: Path | None = None) -> None:
    """Download Oxford-IIIT Pets, build a 50-sample segmentation dataset, and export it."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent

    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir = output_dir / "raw"
    images_dir = output_dir / "images"

    print(f"Downloading Oxford-IIIT Pet dataset to {download_dir} ...")
    train_set = OxfordIIITPet(
        root=str(download_dir),
        split="trainval",
        target_types="segmentation",
        download=True,
    )
    test_set = OxfordIIITPet(
        root=str(download_dir),
        split="test",
        target_types="segmentation",
        download=True,
    )

    print("Selecting samples ...")
    train_items = _select_samples(train_set, TOTAL_TRAIN)
    valtest_items = _select_samples(test_set, TOTAL_VAL + TOTAL_TEST)
    val_items = valtest_items[:TOTAL_VAL]
    test_items = valtest_items[TOTAL_VAL:]
    print(f"  Training   samples: {len(train_items)}")
    print(f"  Validation samples: {len(val_items)}")
    print(f"  Testing    samples: {len(test_items)}")
    print(f"  Total:              {len(train_items) + len(val_items) + len(test_items)}")

    print("Building datumaro experimental dataset with SegmentationSample ...")
    dataset = _build_dataset(train_items, val_items, test_items, images_dir)
    print(f"  Dataset length: {len(dataset)}")

    print(f"Exporting dataset to {output_dir} ...")
    export_dataset(dataset, output_dir)

    print("Cleaning up intermediate files ...")
    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(images_dir, ignore_errors=True)

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a 50-sample Oxford-IIIT Pet segmentation benchmark dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save the dataset. Defaults to tests/assets/segmentation_pets",
    )
    args = parser.parse_args()
    main(output_dir=args.output_dir)
