#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Download and prepare the CarDD instance-segmentation benchmark.

Uses the Kaggle dataset "CarDD" (a COCO-format mirror of *CarDD: A Novel
Dataset for Vision-based Car Damage Detection*, Wang et al.) and exports it in
the experimental Datumaro format, preserving the real per-instance polygon
segmentation masks.

Unlike ``cardd`` (the "CarDD with YOLO annotations" mirror prepared by
``prepare_cardd.py``, which only ships bounding boxes and is therefore a
**detection** benchmark), this Kaggle source carries genuine COCO
``segmentation`` polygons per instance, so it is prepared here as an
**instance segmentation** benchmark — same 6 damage classes (dent, scratch,
crack, glass shatter, lamp broken, tire flat) and same 4,000-image split
sizes (2,816 / 810 / 374) as the detection mirror, since both ultimately
derive from the same upstream CarDD release.

The Kaggle mirror uses a standard COCO "split" layout::

    <root>/train.json
    <root>/val.json
    <root>/test.json
    <root>/train/*.jpg
    <root>/val/*.jpg
    <root>/test/*.jpg

with each ``<split>.json`` holding COCO ``images``/``annotations``/
``categories``, and each annotation's ``segmentation`` a single-ring polygon
(``[[x1, y1, x2, y2, ...]]``).

This dataset is gated behind a Kaggle account. Two ways to provide it:

1. Automatic download — requires benchmark dependencies installed via
    ``just venv-benchmark`` or ``uv sync --extra benchmark`` (from
    ``library/``), and API credentials
   configured via the ``KAGGLE_API_TOKEN`` environment variable, or a
   ``~/.kaggle/access_token`` file. See https://www.kaggle.com/docs/api.
2. Manual placement — download the dataset yourself (e.g. from
   https://www.kaggle.com/datasets/issamjebnouni/cardd) and pass ``--raw-dir
   <path>`` pointing at the extracted dataset root to skip the network step
   entirely while still running through this script's transform/export logic.
"""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

import numpy as np
from datumaro.experimental import Dataset, LazyImage
from datumaro.experimental.data_formats.coco.sample import CocoCategories, CocoSample
from datumaro.experimental.export_import import export_dataset
from datumaro.experimental.fields import ImageInfo, Subset

from getitune.benchmark.dataset_helpers import (
    download_kaggle_dataset,
    parse_args,
    resolve_raw_source,
)

if TYPE_CHECKING:
    from pathlib import Path

_KAGGLE_DATASET_ID = "issamjebnouni/cardd"

# COCO split file/directory name -> Datumaro subset.
_SPLIT_TO_SUBSET = {
    "train": Subset.TRAINING,
    "val": Subset.VALIDATION,
    "test": Subset.TESTING,
}


def _find_coco_root(root: Path) -> Path:
    """Find a CarDD COCO-style export root under *root*.

    "split" layout: ``<root>/<split>.json`` + ``<root>/<split>/`` image
    directory, anchored on the always-present ``train`` split — see
    ``prepare_cardd.py``/``prepare_brain_tumor.py`` for the analogous
    Roboflow/YOLO-layout root finder used by other Kaggle-sourced datasets.
    """
    candidates = sorted(
        candidate for candidate in (path.parent for path in root.rglob("train.json")) if (candidate / "train").is_dir()
    )
    if not candidates:
        msg = f"No CarDD COCO-style dataset root found under {root}"
        raise RuntimeError(msg)
    return candidates[0]


def _polygon_to_array(segmentation: list[list[float]]) -> np.ndarray:
    """Convert a COCO polygon (list of flat ``[x, y, x, y, ...]`` rings) to ``(N, 2)`` float32."""
    # CarDD annotations have a single ring per instance.
    return np.asarray(segmentation[0], dtype=np.float32).reshape(-1, 2)


def _append_image(
    dataset: Dataset,
    img: dict,
    anns: list[dict],
    images_dir: Path,
    subset: Subset,
    cat_id_to_idx: dict[int, int],
) -> None:
    """Build a single ``CocoSample`` (with real polygons) and append it to *dataset*."""
    if anns:
        bboxes = np.asarray([a["bbox"] for a in anns], dtype=np.float32)
        labels = np.asarray([cat_id_to_idx[a["category_id"]] for a in anns], dtype=np.int64)
        areas = np.asarray([a.get("area", 0.0) for a in anns], dtype=np.float32)
        iscrowd = np.asarray([a.get("iscrowd", 0) for a in anns], dtype=np.int32)
        polygon_list = [_polygon_to_array(a["segmentation"]) for a in anns]
        polygons = np.empty(len(polygon_list), dtype=object)
        polygons[:] = polygon_list
    else:
        bboxes = labels = areas = iscrowd = polygons = None

    dataset.append(
        CocoSample(
            image=LazyImage(images_dir / img["file_name"]),
            image_info=ImageInfo(width=int(img["width"]), height=int(img["height"])),
            image_id=img["id"],
            subset=subset,
            bboxes=bboxes,
            labels=labels,
            polygons=polygons,
            areas=areas,
            iscrowd=iscrowd,
            caption_group_ids=None,
            captions=None,
            keypoints=None,
        ),
    )


def _build_dataset(coco_root: Path) -> Dataset:
    """Parse the CarDD COCO instance-segmentation export into a Datumaro dataset."""
    split_data: dict[str, dict] = {}
    for split_name in _SPLIT_TO_SUBSET:
        json_path = coco_root / f"{split_name}.json"
        if json_path.is_file():
            with json_path.open(encoding="utf-8") as f:
                split_data[split_name] = json.load(f)

    if "train" not in split_data:
        msg = f"Expected train.json under CarDD COCO dataset root {coco_root}"
        raise RuntimeError(msg)

    # Categories are identical across all three split files in the upstream
    # CarDD COCO release; read them once from "train" and reuse for the rest.
    categories_sorted = sorted(split_data["train"]["categories"], key=lambda c: c["id"])
    label_names = tuple(c["name"] for c in categories_sorted)
    cat_id_to_idx = {c["id"]: idx for idx, c in enumerate(categories_sorted)}

    dataset: Dataset = Dataset(
        CocoSample,
        categories={"labels": CocoCategories(labels=label_names)},
    )

    for split_name, subset_enum in _SPLIT_TO_SUBSET.items():
        data = split_data.get(split_name)
        if data is None:
            continue
        images_dir = coco_root / split_name

        anns_by_img: dict[int, list[dict]] = {}
        for ann in data["annotations"]:
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

        for img in sorted(data["images"], key=lambda im: im["id"]):
            _append_image(
                dataset,
                img,
                anns_by_img.get(img["id"], []),
                images_dir,
                subset_enum,
                cat_id_to_idx,
            )

    if len(dataset) == 0:
        msg = f"No images found under CarDD COCO dataset root {coco_root}"
        raise RuntimeError(msg)

    return dataset


def main() -> None:
    """Prepare the cardd_instseg benchmark dataset (Kaggle download or --raw-dir) and export it."""
    args = parse_args(description="Prepare the cardd_instseg benchmark dataset.")

    def _download() -> Path:
        return download_kaggle_dataset(_KAGGLE_DATASET_ID)

    raw_root = resolve_raw_source(args, _download)
    dataset = _build_dataset(_find_coco_root(raw_root))

    if args.dest.exists():
        shutil.rmtree(args.dest)
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    export_dataset(dataset, args.dest)

    print(f"Dataset '{args.name}' ready at {args.dest}")


if __name__ == "__main__":
    main()
