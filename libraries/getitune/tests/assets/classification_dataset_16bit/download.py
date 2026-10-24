# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Generate a synthetic 16-bit (uint16) classification dataset in Datumaro experimental format.

The script:
1. Creates synthetic 32x32 RGB images stored as ``uint16`` tensors
   (values in 0-65535 range) with two classes: "circle" and "square".
2. Wraps every image in a ``ClassificationSample``.
3. Stores all samples in a ``datumaro.experimental.Dataset``.
4. Exports the dataset to disk with ``export_dataset``.

This dataset exercises the 16-bit image path (``storage_dtype="uint16"``)
in the data pipeline.

Usage
-----
    python tests/assets/classification_dataset_16bit/download.py [--output_dir OUTPUT_DIR]

The default output directory is
``tests/assets/classification_dataset_16bit``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
import torch
from datumaro.experimental import Dataset
from datumaro.experimental.categories import LabelCategories
from datumaro.experimental.export_import import export_dataset
from datumaro.experimental.fields import ImageInfo as DmImageInfo
from datumaro.experimental.fields import Subset
from datumaro.experimental.fields.images import ImageField
from torchvision import tv_tensors

from getitune.data.entity.sample import ClassificationSample

CLASS_NAMES: tuple[str, ...] = ("circle", "square")
SAMPLES_PER_CLASS_TRAIN: int = 4
SAMPLES_PER_CLASS_VAL: int = 1
SAMPLES_PER_CLASS_TEST: int = 1
IMAGE_SIZE: int = 32


def _make_circle_image(h: int, w: int) -> torch.Tensor:
    """Create a 3xHxW uint16 tensor with a bright circle in the centre."""
    img = torch.zeros(3, h, w, dtype=torch.int32)
    cy, cx = h // 2, w // 2
    r = min(h, w) // 4
    for y in range(h):
        for x in range(w):
            if (y - cy) ** 2 + (x - cx) ** 2 <= r**2:
                img[:, y, x] = 50000  # bright circle on dark background
    return img.to(torch.int32)


def _make_square_image(h: int, w: int) -> torch.Tensor:
    """Create a 3xHxW uint16 tensor with a bright square in the centre."""
    img = torch.zeros(3, h, w, dtype=torch.int32)
    cy, cx = h // 2, w // 2
    half = min(h, w) // 4
    img[:, cy - half : cy + half, cx - half : cx + half] = 40000
    return img.to(torch.int32)


def _generate_image(label_idx: int, seed: int) -> torch.Tensor:
    """Generate a synthetic 3xHxW uint16 image for the given class."""
    torch.manual_seed(seed)
    img = _make_circle_image(IMAGE_SIZE, IMAGE_SIZE) if label_idx == 0 else _make_square_image(IMAGE_SIZE, IMAGE_SIZE)
    # Add slight random noise to make images unique
    noise = torch.randint(0, 1000, (3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.int32)
    return (img + noise).clamp(0, 65535).to(torch.int32)


def _build_dataset() -> Dataset:
    """Build a datumaro.experimental.Dataset of ClassificationSample objects with uint16 images."""
    categories = {"label": LabelCategories(labels=CLASS_NAMES)}

    # Override image field to use UInt16 dtype
    schema = ClassificationSample.infer_schema()
    img_attr = schema.attributes["image"]
    img_attr.field = ImageField(
        semantic="default",
        dtype=pl.UInt16(),
        channels_first=True,
        format="RGB",
    )
    schema.attributes["image"] = img_attr

    dataset: Dataset = Dataset(ClassificationSample, categories=categories, schema=schema)  # type: ignore[arg-type]

    seed = 0
    for subset, count in [
        (Subset.TRAINING, SAMPLES_PER_CLASS_TRAIN),
        (Subset.VALIDATION, SAMPLES_PER_CLASS_VAL),
        (Subset.TESTING, SAMPLES_PER_CLASS_TEST),
    ]:
        for label_idx in range(len(CLASS_NAMES)):
            for _ in range(count):
                img_tensor = _generate_image(label_idx, seed)
                image = tv_tensors.Image(img_tensor)
                sample = ClassificationSample(
                    image=image,
                    label=torch.tensor(label_idx, dtype=torch.uint16),
                    dm_image_info=DmImageInfo(width=IMAGE_SIZE, height=IMAGE_SIZE),
                    subset=subset,
                )
                dataset.append(sample)
                seed += 1

    return dataset


def main(output_dir: Path | None = None) -> None:
    """Generate and export the 16-bit classification dataset."""
    import shutil
    import tempfile

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic 16-bit classification dataset ...")
    dataset = _build_dataset()
    print(f"  Dataset length: {len(dataset)}")

    # export_dataset requires the output dir to not exist,
    # so export to a temp directory first, then move files.
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "export"
        print(f"Exporting dataset to {export_path} ...")
        export_dataset(dataset, export_path)

        # Move exported files to the output directory
        for f in export_path.iterdir():
            dest = output_dir / f.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(f), str(dest))

    print("Done ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic 16-bit classification dataset.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save the dataset. Defaults to tests/assets/classification_dataset_16bit",
    )
    args = parser.parse_args()
    main(output_dir=args.output_dir)
