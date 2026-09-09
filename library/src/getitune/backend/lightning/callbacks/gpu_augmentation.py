# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""GPU Augmentation Callback for applying Kornia augmentations on GPU."""

from __future__ import annotations

import logging as log
import typing
from typing import TYPE_CHECKING

import torch
from lightning import Callback
from torchvision import tv_tensors

from getitune.data.augmentation import GPUAugmentationPipeline
from getitune.data.augmentation.task_keys import DATA_KEYS_BY_TASK
from getitune.data.entity.sample import SampleBatch
from getitune.data.entity.tile import TileBatchData

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer

    from getitune.config.data import SubsetConfig


class GPUAugmentationCallback(Callback):
    """Callback to apply GPU augmentations using Kornia.

    This callback applies GPU-accelerated augmentations from the GPUAugmentationPipeline
    to batches during training and optionally during validation/testing.

    Key features:
    - Applies augmentations after batch is transferred to GPU
    - Extracts normalization parameters and updates model's data_input_params
    - Supports separate train/val pipelines with different augmentation configs
    - Automatically handles bboxes, masks, keypoints based on batch content

    Args:
        train_config: SubsetConfig for training augmentations.
        val_config: SubsetConfig for validation augmentations (optional).
        test_config: SubsetConfig for test augmentations (optional, defaults to val_config).

    Example:
        >>> callback = GPUAugmentationCallback(
        ...     train_config=train_subset_config,
        ...     val_config=val_subset_config,
        ... )
        >>> trainer = Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        train_config: SubsetConfig | None = None,
        val_config: SubsetConfig | None = None,
        test_config: SubsetConfig | None = None,
    ) -> None:
        super().__init__()
        self.train_config = train_config
        self.val_config = val_config
        self.test_config = test_config if test_config is not None else val_config

        self._train_pipeline: GPUAugmentationPipeline | None = None
        self._val_pipeline: GPUAugmentationPipeline | None = None
        self._test_pipeline: GPUAugmentationPipeline | None = None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Setup the GPU augmentation pipelines.

        This is called once when the trainer is setup.
        """
        data_keys = ["input", *DATA_KEYS_BY_TASK.get(pl_module.task, ())]  # type: ignore[arg-type]
        if self.train_config is not None:
            self._train_pipeline = GPUAugmentationPipeline.from_config(self.train_config, data_keys=data_keys)
            log.info(f"GPU train augmentation pipeline:\n{self._train_pipeline}")

        if self.val_config is not None:
            self._val_pipeline = GPUAugmentationPipeline.from_config(
                self.val_config, data_keys=data_keys, sanitize_annotations=False
            )
            log.info(f"GPU val augmentation pipeline:\n{self._val_pipeline}")

        if self.test_config is not None:
            self._test_pipeline = GPUAugmentationPipeline.from_config(
                self.test_config, data_keys=data_keys, sanitize_annotations=False
            )
            log.info(f"GPU test augmentation pipeline:\n{self._test_pipeline}")

        # Update model's normalization params from GPU pipeline
        self._update_model_normalization(pl_module)

    def _update_model_normalization(self, pl_module: LightningModule) -> None:
        """Update model's data_input_params with normalization from GPU pipeline.

        If normalization is in the GPU pipeline, we need to update the model's
        mean/std so that export and inference use the correct values.
        If both model and pipeline have None, set defaults (0,0,0) and (1,1,1).
        """
        # Since we use mean, std values for model export
        # We derive mean, std from test pipeline as priority
        pipeline = self._test_pipeline if self._test_pipeline is not None else self._train_pipeline

        pipeline_mean = pipeline.mean if pipeline is not None else None
        pipeline_std = pipeline.std if pipeline is not None else None
        model_mean = getattr(pl_module.data_input_params, "mean", None)  # type: ignore[union-attr]
        model_std = getattr(pl_module.data_input_params, "std", None)  # type: ignore[union-attr]

        # pipeline > model > default
        data_input_params = pl_module.data_input_params  # type: ignore[union-attr]
        data_input_params.mean = pipeline_mean or model_mean or (0, 0, 0)  # type: ignore[union-attr]
        data_input_params.std = pipeline_std or model_std or (1, 1, 1)  # type: ignore[union-attr]

        # log update
        if model_mean != data_input_params.mean or model_std != data_input_params.std:
            log.info(f"Updated model mean: {model_mean} -> {data_input_params.mean}")
            log.info(f"Updated model std: {model_std} -> {data_input_params.std}")

    def _apply_pipeline(
        self,
        pipeline: GPUAugmentationPipeline,
        batch: SampleBatch | TileBatchData,
    ) -> None:
        """Apply GPU augmentation pipeline to batch (in-place).

        Automatically determines which data to transform based on batch content.
        Kornia decides whether to modify labels based on the augmentations used.

        Args:
            pipeline: GPUAugmentationPipeline to apply.
            batch: SampleBatch (or TileBatchData when tiling is enabled) to transform.
        """
        # Tile batches (produced during validation/testing when tiling is enabled)
        # store their images per-tile in ``batch_tiles`` and keep annotations at the
        # original-image level, so they do not expose ``batch.images``.  Only image
        # transforms such as normalization are meaningful for the individual tiles;
        # their predictions are merged back to the original image afterwards.
        if isinstance(batch, TileBatchData):
            self._apply_pipeline_to_tiles(pipeline, batch)
            return

        # Move pipeline to same device as batch
        device = batch.images.device if hasattr(batch.images, "device") else None
        if device is not None:
            pipeline = pipeline.to(device)

        keypoints_xy: list[torch.Tensor] | None = None
        keypoints_visibility: list[torch.Tensor | None] | None = None
        if batch.keypoints is not None:
            keypoints_xy = []
            keypoints_visibility = []
            _images = typing.cast("torch.Tensor", batch.images)
            for kps in batch.keypoints:
                if kps is None:
                    keypoints_xy.append(torch.empty((0, 2), device=_images.device, dtype=_images.dtype))
                    keypoints_visibility.append(None)
                    continue
                keypoints_xy.append(kps[:, :2])
                if kps.shape[-1] >= 3:
                    keypoints_visibility.append(kps[:, 2])
                else:
                    keypoints_visibility.append(torch.ones(kps.shape[0], device=kps.device, dtype=kps.dtype))

        # Apply pipeline - returns dict with augmented data
        # Labels are included in data_keys, so Kornia will process them if applicable
        result = pipeline(
            batch.images,
            labels=batch.labels,
            bboxes=batch.bboxes,
            masks=batch.masks,
            keypoints=keypoints_xy,
        )

        # Update batch in-place with augmented data
        batch.images = result["images"]
        if result.get("labels") is not None:
            batch.labels = result["labels"]
        if result.get("bboxes") is not None and batch.bboxes is not None:
            # Kornia may return plain tensors, wrap them back to BoundingBoxes
            # Use original canvas_size from batch.bboxes since Kornia does not modify the shape.
            batch.bboxes = [
                tv_tensors.BoundingBoxes(  # type: ignore[no-matching-overload]
                    b,
                    format=tv_tensors.BoundingBoxFormat.XYXY,
                    canvas_size=batch.bboxes[i].canvas_size,
                )
                if not isinstance(b, tv_tensors.BoundingBoxes)
                else b
                for i, b in enumerate(result["bboxes"])
            ]
        if result.get("masks") is not None:
            # Kornia may return plain tensors, wrap them back to Mask
            batch.masks = [tv_tensors.Mask(m) if not isinstance(m, tv_tensors.Mask) else m for m in result["masks"]]
        if result.get("keypoints") is not None:
            if keypoints_visibility is None:
                batch.keypoints = result["keypoints"]
            else:
                # update keypoints visibility based on whether they are in bounds after augmentation
                height, width = batch.images.shape[-2], batch.images.shape[-1]
                restored_keypoints: list[torch.Tensor | None] = []
                for aug_xy, vis in zip(result["keypoints"], keypoints_visibility):
                    if vis is None:
                        restored_keypoints.append(None)
                        continue

                    in_bounds = (
                        (aug_xy[:, 0] >= 0) & (aug_xy[:, 0] < width) & (aug_xy[:, 1] >= 0) & (aug_xy[:, 1] < height)
                    )
                    updated_vis = vis.to(dtype=aug_xy.dtype) * in_bounds.to(dtype=aug_xy.dtype)
                    restored_keypoints.append(torch.cat([aug_xy, updated_vis.unsqueeze(-1)], dim=-1))

                batch.keypoints = typing.cast("list[torch.Tensor] | None", restored_keypoints)

    def _apply_pipeline_to_tiles(
        self,
        pipeline: GPUAugmentationPipeline,
        batch: TileBatchData,
    ) -> None:
        """Apply image-only GPU augmentations (e.g. normalization) to each tile in-place.

        During validation/testing with tiling, the batch is a ``TileBatchData`` whose
        images are stored per-tile in ``batch_tiles`` (annotations are kept at the
        original-image level and used only for tile-merging afterwards). The standard
        ``SampleBatch`` path does not apply here since there is no ``batch.images``.

        Each tile image is passed through the pipeline on its own (as a single-image
        batch). Only image transforms such as normalization are relevant for tiles;
        geometric annotation handling is intentionally skipped because tiles carry no
        per-tile annotations at this stage.

        Args:
            pipeline: GPUAugmentationPipeline to apply.
            batch: TileBatchData whose tiles are transformed in-place.
        """
        for tiles in batch.batch_tiles:
            for tile_idx, tile in enumerate(tiles):
                device = tile.device if hasattr(tile, "device") else None
                tile_pipeline = pipeline.to(device) if device is not None else pipeline
                augmented = tile_pipeline.apply_image_only(tile.unsqueeze(0)).squeeze(0)
                # Re-wrap as an Image tv_tensor so downstream tile handling keeps its type.
                tiles[tile_idx] = tv_tensors.Image(augmented)  # type: ignore[no-matching-overload]

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: SampleBatch,
        batch_idx: int,
    ) -> None:
        """Apply GPU augmentations to training batch."""
        if self._train_pipeline is None:
            return

        self._apply_pipeline(self._train_pipeline, batch)

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: SampleBatch | TileBatchData,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Apply GPU augmentations to validation batch."""
        if self._val_pipeline is None:
            return

        self._apply_pipeline(self._val_pipeline, batch)

    def on_test_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: SampleBatch | TileBatchData,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Apply GPU augmentations to test batch."""
        if self._test_pipeline is None:
            return

        self._apply_pipeline(self._test_pipeline, batch)
