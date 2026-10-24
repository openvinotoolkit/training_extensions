#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Download and prepare the Oxford-IIIT Pet dataset with hierarchical labels.

Exports the dataset with a natural 2-level hierarchy:
  - Level 0 (species): Cat, Dog
  - Level 1 (breeds): 12 cat breeds + 25 dog breeds = 37 leaf classes

This is ideal for the h_label_cls task since the hierarchy is semantically
meaningful (species → breed).

Source: https://www.robots.ox.ac.uk/~vgg/data/pets/
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import numpy as np
from datumaro.experimental import Dataset, LazyImage
from datumaro.experimental.categories import (
    GroupType,
    HierarchicalLabelCategories,
    HierarchicalLabelCategory,
    LabelGroup,
)
from datumaro.experimental.data_formats.coco.sample import CocoSample
from datumaro.experimental.export_import import export_dataset
from datumaro.experimental.fields import ImageInfo, Subset
from PIL import Image

from getitune.benchmark.dataset_helpers import download, extract_archive, parse_args

if TYPE_CHECKING:
    from pathlib import Path

_IMAGES_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
_ANNOTATIONS_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

# Oxford-IIIT Pet breed names (derived from image prefixes).
# Species ID: 1=Cat, 2=Dog (from the annotations file).
# The breed names are extracted from the split files at runtime.


def _parse_splits(splits_dir: Path) -> tuple[dict[str, tuple[int, int]], dict[str, tuple[int, int]]]:
    """Parse trainval.txt and test.txt.

    Returns:
        (trainval_entries, test_entries) where each entry maps
        image_name → (class_id, species_id).
    """

    def _read(filename: str) -> dict[str, tuple[int, int]]:
        result = {}
        path = splits_dir / filename
        if not path.exists():
            return result
        for line in path.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            # Format: image_name class_id species_id breed_id
            name = parts[0]
            class_id = int(parts[1])  # 1-based class (1-37)
            species_id = int(parts[2])  # 1=Cat, 2=Dog
            result[name] = (class_id, species_id)
        return result

    return _read("trainval.txt"), _read("test.txt")


def _build_categories(
    all_entries: dict[str, tuple[int, int]],
) -> tuple[HierarchicalLabelCategories, dict[int, int]]:
    """Build hierarchical categories from the dataset entries.

    Returns:
        (categories, class_id_to_item_idx) mapping class_id (1-37) to item index.
    """
    # Determine breed names from image prefixes grouped by class_id
    class_id_to_names: dict[int, list[str]] = {}
    class_id_to_species: dict[int, int] = {}
    for name, (class_id, species_id) in all_entries.items():
        if class_id not in class_id_to_names:
            class_id_to_names[class_id] = []
            class_id_to_species[class_id] = species_id
        class_id_to_names[class_id].append(name)

    # Derive breed name from the image name prefix (everything before the last _NNN)
    class_id_to_breed: dict[int, str] = {}
    for class_id, names in class_id_to_names.items():
        # All images of the same class share the same prefix
        prefix = "_".join(names[0].rsplit("_", 1)[0].split("_"))
        class_id_to_breed[class_id] = prefix.lower()

    # Build hierarchy items
    items: list[HierarchicalLabelCategory] = []
    # Level 0: species
    items.append(HierarchicalLabelCategory(name="cat", parent=""))
    items.append(HierarchicalLabelCategory(name="dog", parent=""))

    # Level 1: breeds (sorted by class_id for determinism)
    cat_breeds: list[str] = []
    dog_breeds: list[str] = []
    class_id_to_item_idx: dict[int, int] = {}

    for class_id in sorted(class_id_to_breed.keys()):
        breed_name = class_id_to_breed[class_id]
        species = class_id_to_species[class_id]
        parent = "cat" if species == 1 else "dog"
        item_idx = len(items)
        items.append(HierarchicalLabelCategory(name=breed_name, parent=parent))
        class_id_to_item_idx[class_id] = item_idx
        if species == 1:
            cat_breeds.append(breed_name)
        else:
            dog_breeds.append(breed_name)

    # Exclusive label groups
    groups = [
        LabelGroup(name="cat", labels=tuple(cat_breeds), group_type=GroupType.EXCLUSIVE),
        LabelGroup(name="dog", labels=tuple(dog_breeds), group_type=GroupType.EXCLUSIVE),
    ]

    categories = HierarchicalLabelCategories(items=tuple(items), label_groups=tuple(groups))
    return categories, class_id_to_item_idx


def _build_dataset(
    staging: Path,
    categories: HierarchicalLabelCategories,
    class_id_to_item_idx: dict[int, int],
    trainval_entries: dict[str, tuple[int, int]],
    test_entries: dict[str, tuple[int, int]],
) -> Dataset:
    """Build a Datumaro dataset with hierarchical labels."""
    images_dir = staging / "images"

    dataset: Dataset = Dataset(
        CocoSample,
        categories={"labels": categories},
    )

    # Deterministic val split: every 5th trainval image
    trainval_sorted = sorted(trainval_entries.keys())
    val_set = set(trainval_sorted[::5])

    all_images: dict[str, tuple[Subset, int]] = {}
    for name, (class_id, _) in trainval_entries.items():
        subset = Subset.VALIDATION if name in val_set else Subset.TRAINING
        all_images[name] = (subset, class_id)
    for name, (class_id, _) in test_entries.items():
        all_images[name] = (Subset.TESTING, class_id)

    for img_idx, (img_name, (subset, class_id)) in enumerate(sorted(all_images.items())):
        img_path = images_dir / f"{img_name}.jpg"
        if not img_path.exists():
            continue

        label_idx = class_id_to_item_idx[class_id]

        with Image.open(img_path) as im:
            width, height = im.size

        dataset.append(
            CocoSample(
                image=LazyImage(img_path),
                image_info=ImageInfo(width=width, height=height),
                image_id=img_idx,
                subset=subset,
                bboxes=None,
                labels=np.array([label_idx], dtype=np.int64),
                polygons=None,
                areas=None,
                iscrowd=None,
                caption_group_ids=None,
                captions=None,
                keypoints=None,
            ),
        )

    return dataset


def main() -> None:
    """Download Oxford-IIIT Pet, convert to Datumaro format with hierarchical labels, and save."""
    args = parse_args(description="Prepare the oxford_pets_hlabel benchmark dataset.")

    # Download images and annotations
    images_archive = download(_IMAGES_URL, dest_dir=args.archive_dir, filename=f"{args.name}_images.tar.gz")
    ann_archive = download(_ANNOTATIONS_URL, dest_dir=args.archive_dir, filename=f"{args.name}_annotations.tar.gz")

    # Extract into staging
    staging = args.archive_dir / f"{args.name}_raw"
    extract_archive(images_archive, staging, clean_dest=True)
    extract_archive(ann_archive, staging, clean_dest=False)
    images_archive.unlink(missing_ok=True)
    ann_archive.unlink(missing_ok=True)

    # Parse splits
    splits_dir = staging / "annotations"
    trainval_entries, test_entries = _parse_splits(splits_dir)

    # Build categories from all entries
    all_entries = {**trainval_entries, **test_entries}
    categories, class_id_to_item_idx = _build_categories(all_entries)

    print("Building Datumaro dataset from Oxford-IIIT Pet (hierarchical labels) ...")
    dataset = _build_dataset(staging, categories, class_id_to_item_idx, trainval_entries, test_entries)
    print(f"  Dataset length: {len(dataset)}")

    if args.dest.exists():
        shutil.rmtree(args.dest)
    args.dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting dataset to {args.dest} ...")
    export_dataset(dataset, args.dest)

    # Cleanup
    shutil.rmtree(staging, ignore_errors=True)

    print(f"Dataset '{args.name}' ready at {args.dest}")


if __name__ == "__main__":
    main()
