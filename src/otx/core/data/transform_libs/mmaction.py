# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Helper to support MMAction data transform functions."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import mmcv
import numpy as np
from mmaction.datasets.transforms import PackActionInputs as MMPackActionInputs
from mmaction.datasets.transforms import RawFrameDecode as MMRawFrameDecode
from mmaction.registry import TRANSFORMS
from mmengine.fileio import FileClient
from torchvision import tv_tensors

from otx.core.data.entity.action_classification import ActionClsDataEntity
from otx.core.data.entity.action_detection import ActionDetDataEntity
from otx.core.utils.config import convert_conf_to_mmconfig_dict

if TYPE_CHECKING:
    from mmengine.registry import Registry

    from otx.core.config.data import SubsetConfig


@TRANSFORMS.register_module()
class LoadVideoForClassification:
    """Class to convert OTXDataEntity to dict for MMAction framework."""

    def __call__(self, entity: ActionClsDataEntity) -> dict:
        """Transform ActionClsDataEntity to MMAction data dictionary format."""
        results: dict[str, Any] = {}
        results["filename"] = entity.video.path
        results["start_index"] = 0
        results["modality"] = "RGB"
        results["__otx__"] = entity

        return results


@TRANSFORMS.register_module()
class LoadVideoForDetection:
    """Class to convert OTXDataEntity to dict for MMAction framework."""

    fps: int = 30
    timestamp_start: int = 900

    def __call__(self, entity: ActionDetDataEntity) -> dict:
        """Transform ActionClsDataEntity to MMAction data dictionary format."""
        results: dict[str, Any] = {}
        results["modality"] = "RGB"
        results["fps"] = self.fps
        results["timestamp_start"] = self.timestamp_start
        results["filename_tmpl"] = "{video}_{idx:04d}.{ext}"
        results["ori_shape"] = entity.img_info.ori_shape

        if entity.frame_path is not None:
            frame_dir, extension, shot_info, timestamp = self._get_meta_info(entity.frame_path)
            results["timestamp"] = timestamp
            results["shot_info"] = shot_info
            results["frame_dir"] = frame_dir
            results["extension"] = extension

        results["__otx__"] = entity

        return results

    @staticmethod
    def _get_meta_info(frame_path: str) -> tuple[Path, str, tuple[int, int], int]:
        frame_dir = Path(frame_path).parent
        extension = Path(frame_path).suffix[1:]
        shot_info = (1, len(os.listdir(frame_dir)))
        timestamp = int(Path(frame_path).stem.split("_")[-1])

        return frame_dir, extension, shot_info, timestamp


@TRANSFORMS.register_module()
class LoadAnnotations:
    """Load annotation infomation such as ground truth bounding boxes and proposals."""

    def __call__(self, results: dict) -> dict:
        """Get ground truth information from data entity."""
        if (otx_data_entity := results.get("__otx__")) is None:
            msg = "__otx__ key should be passed from the previous pipeline (LoadImageFromFile)"
            raise RuntimeError(msg)

        results["gt_bboxes"] = otx_data_entity.bboxes.numpy()
        results["gt_labels"] = otx_data_entity.labels.numpy()
        results["proposals"] = otx_data_entity.proposals

        return results


@TRANSFORMS.register_module(force=True)
class RawFrameDecode(MMRawFrameDecode):
    """Load and decode frames with given indices.

    This Custom RawFrameDecode pipeline is for Datumaro ava dataset format.
    """

    file_client: FileClient | None

    def transform(self, results: dict) -> dict:
        """Perform the ``RawFrameDecode`` to pick frames given indices.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        mmcv.use_backend(self.decoding_backend)

        directory = results["frame_dir"]
        filename_tmpl = results["filename_tmpl"]
        video = Path(results["frame_dir"]).name
        ext = results["extension"]

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        imgs: list[np.ndarray] = []

        if results["frame_inds"].ndim != 1:
            results["frame_inds"] = np.squeeze(results["frame_inds"])

        offset = results.get("offset", 0)

        cache: dict[int, int] = {}
        for i, frame_idx in enumerate(results["frame_inds"]):
            # Avoid loading duplicated frames
            if frame_idx in cache:
                imgs.append(deepcopy(imgs[cache[frame_idx]]))
                continue
            cache[frame_idx] = i

            frame_idx_with_offset = frame_idx + offset
            filepath = Path(directory) / filename_tmpl.format(video=video, idx=frame_idx_with_offset, ext=ext)
            img_bytes = self.file_client.get(filepath)
            # Get frame with channel order RGB directly.
            cur_frame = mmcv.imfrombytes(img_bytes, channel_order="rgb")
            imgs.append(cur_frame)

        results["imgs"] = imgs
        results["original_shape"] = imgs[0].shape[:2]
        results["img_shape"] = imgs[0].shape[:2]

        # we resize the gt_bboxes and proposals to their real scale
        if "gt_bboxes" in results:
            h, w = results["img_shape"]
            scale_factor = np.array([w, h, w, h])
            gt_bboxes = results["gt_bboxes"]
            gt_bboxes = (gt_bboxes * scale_factor).astype(np.float32)
            results["gt_bboxes"] = gt_bboxes
            if "proposals" in results and results["proposals"] is not None:
                proposals = results["proposals"]
                proposals = (proposals * scale_factor).astype(np.float32)
                results["proposals"] = proposals

        return results


@TRANSFORMS.register_module(force=True)
class PackActionInputs(MMPackActionInputs):
    """Class to override PackActionInputs.

    Transfrom output dictionary from MMAction to ActionClsDataEntity or ActionDetDataEntity.
    """

    def transform(self, results: dict) -> ActionClsDataEntity | ActionDetDataEntity:
        """Transform function."""
        transformed = super().transform(results)
        image = tv_tensors.Image(transformed.get("inputs"))
        data_samples = transformed["data_samples"]

        ori_shape = results["original_shape"]
        img_shape = data_samples.img_shape
        scale_factor = data_samples.metainfo.get("scale_factor", (1.0, 1.0))

        data_entity: ActionClsDataEntity | ActionDetDataEntity = results["__otx__"]

        image_info = deepcopy(data_entity.img_info)
        image_info.img_shape = img_shape
        image_info.ori_shape = ori_shape
        image_info.scale_factor = scale_factor

        labels = data_entity.labels

        if "gt_bboxes" in results:
            proposals = tv_tensors.BoundingBoxes(
                data_samples.proposals.bboxes.float(),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=img_shape,
            )
            bboxes = tv_tensors.BoundingBoxes(
                data_samples.gt_instances.bboxes.float(),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=img_shape,
            )

            return ActionDetDataEntity(
                image=image,
                img_info=image_info,
                bboxes=bboxes,
                labels=labels,
                proposals=proposals,
                frame_path=results["__otx__"].frame_path,
            )

        return ActionClsDataEntity(
            video=results["__otx__"].video,
            image=image,
            img_info=image_info,
            labels=labels,
        )


class MMActionTransformLib:
    """Helper to support MMCV transforms in OTX."""

    @classmethod
    def get_builder(cls) -> Registry:
        """Transform builder obtained from MMCV."""
        return TRANSFORMS

    @classmethod
    def _check_mandatory_transforms(
        cls,
        transforms: list[Callable],
        mandatory_transforms: set,
    ) -> None:
        for transform in transforms:
            t_transform = type(transform)
            mandatory_transforms.discard(t_transform)

        if len(mandatory_transforms) != 0:
            msg = f"{mandatory_transforms} should be included"
            raise RuntimeError(msg)

    @classmethod
    def generate(cls, config: SubsetConfig) -> list[Callable]:
        """Generate MMCV transforms from the configuration."""
        return [cls.get_builder().build(convert_conf_to_mmconfig_dict(cfg)) for cfg in config.transforms]
