# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared helper for HuggingFace "image + label-mask pair" semantic-segmentation mirrors.

A handful of public HuggingFace datasets ship real per-pixel semantic-segmentation
masks as a simple two-column parquet schema::

    <image_column> : struct<bytes: binary, path: string>   # embedded RGB photo
    <mask_column>  : struct<bytes: binary, path: string>   # embedded label mask image

Unlike the (bbox/polygon) Roboflow mirrors handled by ``roboflow_hf_helper.py``, the
mask encoding here is *not* standardized: some datasets store a near-binary grayscale
mask (with antialiasing noise baked in from an upstream resize step), others store an
RGB, VOC-style palette with one flat colour per class. Each ``prepare_*.py`` script is
therefore expected to supply a small ``mask_decoder`` callback that turns a raw, decoded
``PIL.Image`` into a clean ``(H, W)`` ``uint8`` array of class indices (index 0 =
background/void), and this module takes care of the download, staging, and Datumaro
dataset construction/export around it.

Images and masks are written to a staging directory and referenced lazily (``LazyImage``
/ a small loader callable) rather than embedded eagerly, so datasets with thousands of
images can be built without holding every image in memory at once — the same pattern
``roboflow_hf_helper.py`` uses for images.
"""

from __future__ import annotations

import functools
import io
import logging
import shutil
from typing import TYPE_CHECKING, Callable

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from datumaro.experimental import Dataset, LazyImage, register_sample
from datumaro.experimental.categories import MaskCategories
from datumaro.experimental.dataset import Sample
from datumaro.experimental.export_import import export_dataset
from datumaro.experimental.fields import (
    ImageInfo,
    Subset,
    image_info_field,
    image_path_field,
    mask_callable_field,
    subset_field,
)
from PIL import Image as PILImage

from getitune.benchmark.dataset_helpers import download

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from getitune.benchmark.dataset_helpers import DatasetArgs

logger = logging.getLogger(__name__)

# Rows read from each parquet at a time. Image+mask pairs are heavier than the
# bbox-only rows handled by ``roboflow_hf_helper.py``, so a smaller batch keeps peak
# memory bounded.
_BATCH_SIZE = 64

# ``path`` is a repo-relative path, e.g. ``data/train-x.parquet``.
_HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"


@register_sample
class _MaskPrepSample(Sample):
    """Lazy Datumaro sample used only while preparing semantic-segmentation datasets.

    Mirrors ``getitune.data.entity.sample.SegmentationSample`` but keeps the image and
    mask lazily loaded from disk (``LazyImage`` / a loader callable) instead of eagerly
    embedding pixel data. ``mask`` is declared with the default ("default") semantic
    (like ``SegmentationSample.masks``), so ``convert_to_schema(SegmentationSample)``
    resolves it directly at training time without any bbox/polygon involved.
    """

    image: LazyImage = image_path_field()
    image_info: ImageInfo = image_info_field()
    mask: Callable[[], np.ndarray] | None = mask_callable_field(dtype=pl.UInt8())
    subset: Subset = subset_field()


def _load_mask_png(path: Path) -> np.ndarray:
    """Lazily load a previously-decoded, single-channel class-index mask PNG."""
    return np.array(PILImage.open(path))


def prepare_hf_mask_dataset(
    args: DatasetArgs,
    *,
    repo: str,
    revision: str,
    split_files: Mapping[str, str | Sequence[str]],
    label_names: Sequence[str],
    image_column: str,
    mask_column: str,
    mask_decoder: Callable[[PILImage.Image], np.ndarray],
    assign_subset: Callable[[str, int], Subset],
) -> None:
    """Download an image/mask-pair HuggingFace mirror and export a Datumaro dataset.

    Args:
        args: Parsed CLI arguments (``--output-dir`` / ``--name``).
        repo: HuggingFace dataset id, e.g. ``"varcoder/crack-segmentation-dataset"``.
        revision: Pinned commit SHA of the dataset for reproducibility.
        split_files: Mapping of upstream split name to one or more repo-relative
            parquet paths (some splits are sharded across several parquet files).
        label_names: Class names in class-index order (index 0 is conventionally
            background/void).
        image_column: Name of the HF ``Image`` column holding the RGB photo.
        mask_column: Name of the HF ``Image`` column holding the raw label mask.
        mask_decoder: Converts a raw, decoded ``PIL.Image`` mask into a final
            ``(H, W)`` ``uint8`` class-index array (e.g. thresholding a near-binary
            mask, or mapping an RGB palette to class indices).
        assign_subset: Called once per row, in iteration order (splits in
            *split_files* dict order, rows within a split in parquet row order,
            shards within a split in list order), with ``(upstream_split_name,
            global_row_index)``. Must return the Datumaro ``Subset`` to assign that
            row to. This indirection lets callers re-split datasets that do not ship
            a full train/validation/test triple upstream (e.g. carving a
            validation set out of train, or merging+re-splitting from scratch).
    """
    dataset: Dataset = Dataset(
        _MaskPrepSample,
        categories={"mask": MaskCategories(labels=list(label_names))},
    )

    staging = args.archive_dir / f"{args.name}_raw"
    images_dir = staging / "images"
    masks_dir = staging / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    downloaded_parquets: list[Path] = []
    row_id = 0

    for split_name, raw_paths in split_files.items():
        paths = [raw_paths] if isinstance(raw_paths, str) else list(raw_paths)
        for shard_idx, path in enumerate(paths):
            url = _HF_RESOLVE.format(repo=repo, revision=revision, path=path)
            parquet_path = download(
                url,
                dest_dir=args.archive_dir,
                filename=f"{args.name}_{split_name}_{shard_idx}.parquet",
            )
            downloaded_parquets.append(parquet_path)

            parquet_file = pq.ParquetFile(str(parquet_path))
            for batch in parquet_file.iter_batches(batch_size=_BATCH_SIZE):
                images = batch.column(image_column).to_pylist()
                masks = batch.column(mask_column).to_pylist()

                for image_struct, mask_struct in zip(images, masks, strict=True):
                    subset = assign_subset(split_name, row_id)

                    img_bytes = image_struct["bytes"]
                    img_suffix = (image_struct.get("path") or "image.jpg").rsplit(".", 1)[-1].lower()
                    img_path = images_dir / f"img_{row_id:06d}.{img_suffix}"
                    img_path.write_bytes(img_bytes)

                    with PILImage.open(io.BytesIO(mask_struct["bytes"])) as raw_mask:
                        decoded_mask = np.asarray(mask_decoder(raw_mask), dtype=np.uint8)
                    if decoded_mask.ndim != 2:
                        msg = f"mask_decoder must return a 2D (H, W) array, got shape {decoded_mask.shape}"
                        raise ValueError(msg)
                    height, width = decoded_mask.shape

                    mask_path = masks_dir / f"mask_{row_id:06d}.png"
                    PILImage.fromarray(decoded_mask, mode="L").save(mask_path)

                    dataset.append(
                        _MaskPrepSample(
                            image=LazyImage(img_path),
                            image_info=ImageInfo(width=int(width), height=int(height)),
                            mask=functools.partial(_load_mask_png, mask_path),
                            subset=subset,
                        ),
                    )
                    row_id += 1

    print(f"  Dataset length: {len(dataset)}")

    # ``export_dataset`` requires that the output path does NOT exist yet,
    # so remove any leftover from a previous run and let it create the dir.
    if args.dest.exists():
        shutil.rmtree(args.dest)
    args.dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting dataset to {args.dest} ...")
    export_dataset(dataset, args.dest)

    # Clean up the downloaded parquet splits and staged images/masks.
    for parquet_path in downloaded_parquets:
        parquet_path.unlink(missing_ok=True)
    shutil.rmtree(staging, ignore_errors=True)

    print(f"Dataset '{args.name}' ready at {args.dest}")
