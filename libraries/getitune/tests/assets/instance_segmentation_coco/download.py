# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Download a ~50-sample COCO 2017 subset for instance segmentation.

Downloads the COCO 2017 val annotations and a subset of images, then creates
``InstanceSegmentationSample`` objects with bounding boxes and instance masks.

Usage
-----
    python tests/assets/instance_segmentation_coco/download.py [--output_dir OUTPUT_DIR]

Requirements
------------
``pycocotools`` must be installed (``pip install pycocotools``).
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from datumaro.experimental import Dataset
from datumaro.experimental.categories import Colormap, LabelCategories, MaskCategories, RgbColor
from datumaro.experimental.export_import import export_dataset
from datumaro.experimental.fields import ImageInfo as DmImageInfo
from datumaro.experimental.fields import Subset
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from torchvision import tv_tensors

from getitune.data.entity.sample import InstanceSegmentationSample

COCO_ANNOTATIONS_URL = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_VAL_IMAGES_URL = "https://images.cocodataset.org/val2017/{filename}"

SAMPLES_TRAIN: int = 35
SAMPLES_VAL: int = 8
SAMPLES_TEST: int = 7

# Resize images to keep the exported dataset under 3 MB
IMG_SIZE: int = 64


def _download_annotations(download_dir: Path) -> Path:
    """Download and extract COCO 2017 annotations if not already present."""
    ann_file = download_dir / "annotations" / "instances_val2017.json"
    if ann_file.exists():
        print("  Annotations already downloaded.")
        return ann_file

    zip_path = download_dir / "annotations_trainval2017.zip"
    if not zip_path.exists():
        print(f"  Downloading annotations from {COCO_ANNOTATIONS_URL} ...")
        download_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(COCO_ANNOTATIONS_URL, zip_path)  # noqa: S310

    print("  Extracting annotations ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(download_dir)

    return ann_file


def _download_image(image_filename: str, download_dir: Path) -> Path:
    """Download a single COCO image if not already present."""
    img_path = download_dir / image_filename
    if not img_path.exists():
        url = COCO_VAL_IMAGES_URL.format(filename=image_filename)
        urllib.request.urlretrieve(url, img_path)  # noqa: S310
    return img_path


def _select_images(
    coco: COCO,
    total: int,
) -> list[int]:
    """Select image IDs that have at least one instance annotation with a polygon mask."""
    selected: list[int] = []
    for img_id in sorted(coco.getImgIds()):
        if len(selected) >= total:
            break
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        # Only keep images with at least one polygon segmentation
        valid = [a for a in anns if a.get("segmentation") and isinstance(a["segmentation"], list)]
        if valid:
            selected.append(img_id)
    return selected


def _build_dataset(
    coco: COCO,
    image_ids: list[int],
    images_dir: Path,
    subset: Subset,
    cat_id_to_idx: dict[int, int],
) -> list[InstanceSegmentationSample]:
    """Create InstanceSegmentationSample objects from COCO annotations."""
    from PIL import Image as PILImage

    samples: list[InstanceSegmentationSample] = []

    for img_id in image_ids:
        img_info = coco.loadImgs(img_id)[0]
        filename = img_info["file_name"]
        w, h = img_info["width"], img_info["height"]

        # Download image
        img_path = _download_image(filename, images_dir)
        img_pil = PILImage.open(img_path).convert("RGB")
        img_pil = img_pil.resize((IMG_SIZE, IMG_SIZE), PILImage.BILINEAR)  # type: ignore[attr-defined]
        img_np = np.array(img_pil, dtype=np.uint8)
        img_chw = torch.from_numpy(img_np.transpose(2, 0, 1))
        image = tv_tensors.Image(img_chw)

        # Parse annotations
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        valid_anns = [a for a in anns if a.get("segmentation") and isinstance(a["segmentation"], list)]

        bboxes_list = []
        labels_list = []
        masks_list = []

        sx, sy = IMG_SIZE / w, IMG_SIZE / h
        for ann in valid_anns:
            if ann["category_id"] not in cat_id_to_idx:
                continue
            # COCO bbox is [x, y, w, h] -> convert to [x1, y1, x2, y2]
            bx, by, bw, bh = ann["bbox"]
            bboxes_list.append([bx * sx, by * sy, (bx + bw) * sx, (by + bh) * sy])
            labels_list.append(cat_id_to_idx[ann["category_id"]])

            # Decode polygon segmentation to binary mask at original size, then resize
            rle = mask_utils.frPyObjects(ann["segmentation"], h, w)  # type: ignore[arg-type]
            rle = mask_utils.merge(rle)
            binary_mask = mask_utils.decode(rle)
            mask_pil = PILImage.fromarray(binary_mask)
            mask_pil = mask_pil.resize((IMG_SIZE, IMG_SIZE), PILImage.NEAREST)  # type: ignore[attr-defined]
            masks_list.append(np.array(mask_pil, dtype=np.uint8))  # type: ignore[arg-type]

        if not bboxes_list:
            continue

        bboxes = torch.tensor(bboxes_list, dtype=torch.float32)
        labels = torch.tensor(labels_list, dtype=torch.long)
        masks = tv_tensors.Mask(torch.from_numpy(np.stack(masks_list, axis=0).astype(np.uint8)))  # type: ignore[arg-type]

        sample = InstanceSegmentationSample(
            image=image,
            bboxes=bboxes,
            masks=masks,
            label=labels,
            dm_image_info=DmImageInfo(width=IMG_SIZE, height=IMG_SIZE),
            subset=subset,
        )
        samples.append(sample)

    return samples


def main(output_dir: Path | None = None) -> None:
    """Download COCO 2017 subset, build a 50-sample instance segmentation dataset, and export it."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent

    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir = output_dir / "raw"
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("Setting up COCO annotations ...")
    ann_file = _download_annotations(download_dir)

    print("Loading COCO annotations ...")
    coco = COCO(str(ann_file))

    print("Selecting images ...")
    all_ids = _select_images(coco, SAMPLES_TRAIN + SAMPLES_VAL + SAMPLES_TEST)
    train_ids = all_ids[:SAMPLES_TRAIN]
    val_ids = all_ids[SAMPLES_TRAIN : SAMPLES_TRAIN + SAMPLES_VAL]
    test_ids = all_ids[SAMPLES_TRAIN + SAMPLES_VAL :]
    print(f"  Training   images: {len(train_ids)}")
    print(f"  Validation images: {len(val_ids)}")
    print(f"  Testing    images: {len(test_ids)}")

    # Collect only the category IDs that actually appear in the selected images.
    used_cat_ids: set[int] = set()
    for img_id in all_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        for ann in coco.loadAnns(ann_ids):
            used_cat_ids.add(ann["category_id"])
    used_cat_ids_sorted = sorted(used_cat_ids)
    cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(used_cat_ids_sorted)}
    used_cats = coco.loadCats(used_cat_ids_sorted)
    cat_names = tuple(c["name"] for c in used_cats)
    print(f"  {len(cat_names)} categories used (of {len(coco.getCatIds())} total)")

    print("Downloading images and building dataset ...")
    # Instance masks are binary (0 = background, 1 = instance), need MaskCategories
    mask_colormap = Colormap({0: RgbColor(0, 0, 0), 1: RgbColor(255, 255, 255)})
    categories = {
        "label": LabelCategories(labels=cat_names),
        "masks": MaskCategories(labels=["background", "instance"], colormap=mask_colormap),
    }
    dataset: Dataset = Dataset(InstanceSegmentationSample, categories=categories)  # type: ignore[arg-type]

    for sample in _build_dataset(coco, train_ids, images_dir, Subset.TRAINING, cat_id_to_idx):
        dataset.append(sample)
    for sample in _build_dataset(coco, val_ids, images_dir, Subset.VALIDATION, cat_id_to_idx):
        dataset.append(sample)
    for sample in _build_dataset(coco, test_ids, images_dir, Subset.TESTING, cat_id_to_idx):
        dataset.append(sample)

    print(f"  Dataset length: {len(dataset)}")

    print(f"Exporting dataset to {output_dir} ...")
    export_dataset(dataset, output_dir)

    print("Cleaning up intermediate files ...")
    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(images_dir, ignore_errors=True)

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a 50-sample COCO 2017 instance segmentation benchmark dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save the dataset. Defaults to tests/assets/instance_segmentation_coco",
    )
    args = parser.parse_args()
    main(output_dir=args.output_dir)
