# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Download a ~50-sample COCO 2017 subset for keypoint detection.

Downloads COCO 2017 val annotations and a subset of images, then creates
``KeypointSample`` objects with person keypoints (17 keypoints per instance).
Each sample corresponds to a single person annotation (cropped context).

Usage
-----
    python tests/assets/keypoint_detection_coco/download.py [--output_dir OUTPUT_DIR]

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
from datumaro.experimental.categories import KeypointCategories, LabelCategories
from datumaro.experimental.export_import import export_dataset
from datumaro.experimental.fields import ImageInfo as DmImageInfo
from datumaro.experimental.fields import Subset
from pycocotools.coco import COCO
from torchvision import tv_tensors

from getitune.data.entity.sample import KeypointSample

COCO_ANNOTATIONS_URL = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_VAL_IMAGES_URL = "https://images.cocodataset.org/val2017/{filename}"

# COCO person keypoint names
COCO_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

SAMPLES_TRAIN: int = 35
SAMPLES_VAL: int = 8
SAMPLES_TEST: int = 7

# Resize images to keep the exported dataset under 3 MB
IMG_SIZE: int = 64


def _download_annotations(download_dir: Path) -> Path:
    """Download and extract COCO 2017 annotations if not already present."""
    ann_file = download_dir / "annotations" / "person_keypoints_val2017.json"
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


def _collect_person_annotations(
    coco: COCO,
    total: int,
) -> list[tuple[dict, dict]]:
    """Collect person annotations that have enough visible keypoints.

    Returns a list of (annotation_dict, image_info_dict) tuples.
    """
    person_cat_ids = coco.getCatIds(catNms=["person"])
    selected: list[tuple[dict, dict]] = []

    for img_id in sorted(coco.getImgIds(catIds=person_cat_ids)):
        if len(selected) >= total:
            break
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=person_cat_ids, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs(img_id)[0]

        for ann in anns:
            if len(selected) >= total:
                break
            kps = ann.get("keypoints", [])
            if len(kps) != 17 * 3:  # type: ignore[arg-type]
                continue
            kps_array = np.array(kps, dtype=np.float32).reshape(17, 3)
            # At least 5 visible keypoints
            num_visible = int((kps_array[:, 2] > 0).sum())
            if num_visible >= 5:
                selected.append((ann, img_info))  # type: ignore[arg-type]

    return selected


def _build_dataset(
    annotations: list[tuple[dict, dict]],
    images_dir: Path,
    subset: Subset,
) -> list[KeypointSample]:
    """Create KeypointSample objects from COCO keypoint annotations."""
    from PIL import Image as PILImage

    samples: list[KeypointSample] = []
    downloaded_images: dict[str, tuple[np.ndarray, int, int]] = {}

    for ann, img_info in annotations:
        filename = img_info["file_name"]
        _w, _h = img_info["width"], img_info["height"]

        # Download and cache image
        if filename not in downloaded_images:
            img_path = _download_image(filename, images_dir)
            img_pil = PILImage.open(img_path).convert("RGB")
            orig_w, orig_h = img_pil.size
            img_pil = img_pil.resize((IMG_SIZE, IMG_SIZE), PILImage.BILINEAR)  # type: ignore[attr-defined]
            downloaded_images[filename] = (np.array(img_pil, dtype=np.uint8), orig_w, orig_h)

        img_np, orig_w, orig_h = downloaded_images[filename]
        img_chw = torch.from_numpy(img_np.transpose(2, 0, 1))
        image = tv_tensors.Image(img_chw)

        # Keypoints: (17, 3) with [x, y, visibility] — scale to resized image
        kps = np.array(ann["keypoints"], dtype=np.float32).reshape(17, 3)
        kps[:, 0] *= IMG_SIZE / orig_w
        kps[:, 1] *= IMG_SIZE / orig_h
        keypoints = torch.from_numpy(kps)

        # Label: person = 0 (single class)
        label = torch.tensor([0], dtype=torch.long)

        sample = KeypointSample(
            image=image,
            label=label,
            keypoints=keypoints,
            dm_image_info=DmImageInfo(width=IMG_SIZE, height=IMG_SIZE),
            subset=subset,
        )
        samples.append(sample)

    return samples


def main(output_dir: Path | None = None) -> None:
    """Download COCO 2017 subset, build a 50-sample keypoint dataset, and export it."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent

    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir = output_dir / "raw"
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("Setting up COCO annotations ...")
    ann_file = _download_annotations(download_dir)

    print("Loading COCO person_keypoints annotations ...")
    coco = COCO(str(ann_file))

    print("Collecting person annotations with keypoints ...")
    all_anns = _collect_person_annotations(coco, SAMPLES_TRAIN + SAMPLES_VAL + SAMPLES_TEST)
    train_anns = all_anns[:SAMPLES_TRAIN]
    val_anns = all_anns[SAMPLES_TRAIN : SAMPLES_TRAIN + SAMPLES_VAL]
    test_anns = all_anns[SAMPLES_TRAIN + SAMPLES_VAL :]
    print(f"  Training   samples: {len(train_anns)}")
    print(f"  Validation samples: {len(val_anns)}")
    print(f"  Testing    samples: {len(test_anns)}")

    print("Downloading images and building dataset ...")
    categories = {
        "label": LabelCategories(labels=("person",)),
        "keypoints": KeypointCategories(labels=COCO_KEYPOINT_NAMES),
    }
    dataset: Dataset = Dataset(KeypointSample, categories=categories)  # type: ignore[arg-type]

    for sample in _build_dataset(train_anns, images_dir, Subset.TRAINING):
        dataset.append(sample)
    for sample in _build_dataset(val_anns, images_dir, Subset.VALIDATION):
        dataset.append(sample)
    for sample in _build_dataset(test_anns, images_dir, Subset.TESTING):
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
        description="Download a 50-sample COCO 2017 keypoint detection benchmark dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save the dataset. Defaults to tests/assets/keypoint_detection_coco",
    )
    args = parser.parse_args()
    main(output_dir=args.output_dir)
