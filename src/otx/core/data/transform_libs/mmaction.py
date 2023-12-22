# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Helper to support MMAction data transform functions."""

from __future__ import annotations

import os
import pickle
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import mmcv
import numpy as np
from mmaction.datasets.transforms import PackActionInputs as MMPackActionInputs
from mmaction.datasets.transforms import RawFrameDecode as MMRawFrameDecode
from mmaction.datasets.transforms import SampleFrames as MMSampleFrames
from mmaction.registry import TRANSFORMS
from mmengine.fileio import FileClient
from torchvision import tv_tensors

from otx.core.data.entity.action_classification import ActionClsDataEntity
from otx.core.data.entity.action_detection import ActionDetDataEntity
from otx.core.data.entity.base import ImageInfo
from otx.core.utils.config import convert_conf_to_mmconfig_dict

if TYPE_CHECKING:
    from mmengine.registry import Registry

    from otx.core.config.data import SubsetConfig


# @TRANSFORMS.register_module(force=True)
# class LoadVideo:
#     """Class to convert OTXDataEntity to dict for MMAction framework."""

#     def __call__(self, entity: ActionClsDataEntity) -> dict:
#         """Transform ActionClsDataEntity to MMAction data dictionary format."""
#         video: list[np.ndarray] = entity.image

#         results: dict[str, Any] = {}
#         results["filename"] = entity.video.path
#         results["start_index"] = 0
#         results["modality"] = "RGB"
#         results["__otx__"] = entity

#         return results


@TRANSFORMS.register_module(force=True)
class LoadVideo:
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
        results["proposal_file"] = entity.proposal_file
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


@TRANSFORMS.register_module(force=True)
class LoadAnnotations:
    """Load annotation infomation such as ground truth bounding boxes and proposals."""

    def __call__(self, results: dict) -> dict:
        """Get ground truth information from data entity."""
        if (otx_data_entity := results.get("__otx__")) is None:
            msg = "__otx__ key should be passed from the previous pipeline (LoadImageFromFile)"
            raise RuntimeError(msg)

        results["gt_bboxes"] = otx_data_entity.bboxes.numpy()
        results["gt_labels"] = otx_data_entity.labels.numpy()
        results["proposals"] = self._get_proposals(otx_data_entity.frame_path, results["proposal_file"])

        return results

    @staticmethod
    def _get_proposals(frame_path: str, proposal_file: str | None) -> np.ndarray:
        """Get proposal from frame path and proposal file name.

        Datumaro AVA dataset expect data structure as
        - data_root/
            - frames/
                - video0
                    - video0_0001.jpg
                    - vdieo0_0002.jpg
            - annotations/
                - train.csv
                - val.csv
                - train.pkl
                - val.pkl
        """
        if proposal_file is None:
            return np.array([[0, 0, 1, 1]], dtype=np.float64)

        annotation_dir = Path(frame_path).parent.parent.parent
        proposal_file_path = annotation_dir / "annotations" / proposal_file
        if proposal_file_path.exists():
            with Path.open(proposal_file_path, "rb") as f:
                info = pickle.load(f)  # noqa: S301
                if Path(frame_path).stem.replace("_", ",") in info:
                    proposals = info[Path(frame_path).stem.replace("_", ",")][:, :4]
                else:
                    proposals = np.array([[0, 0, 1, 1]], dtype=np.float64)
        else:
            proposals = np.array([[0, 0, 1, 1]], dtype=np.float64)

        return proposals


@TRANSFORMS.register_module(force=True)
class SampleFrames(MMSampleFrames):
    """Class to override SampleFrames.

    MMAction's SampleFrames just sample frame indices for training.
    Actual frame sampling is done by decode pipeline.
    However, OTX already has decoded data, so here, actual sampling frame will be conducted.
    """

    def transform(self, results: dict) -> dict:
        """Transform function."""
        super().transform(results)
        imgs: list[np.ndarray] = [results["imgs"][idx] for idx in results["frame_inds"]]
        results["imgs"] = imgs

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
class PackActionClsInputs(MMPackActionInputs):
    """Class to override PackActionInputs.

    Transfrom output dictionary from MMAction to ActionClsDataEntity.
    """

    def transform(self, results: dict) -> ActionClsDataEntity:
        """Transform function."""
        transformed = super().transform(results)
        image = tv_tensors.Image(transformed.get("inputs"))
        data_samples = transformed["data_samples"]

        ori_shape = results["original_shape"]
        img_shape = data_samples.img_shape
        pad_shape = data_samples.metainfo.get("pad_shape", img_shape)
        scale_factor = data_samples.metainfo.get("scale_factor", (1.0, 1.0))

        labels = results["__otx__"].labels

        return ActionClsDataEntity(
            video=results["__otx__"].video,
            image=image,
            img_info=ImageInfo(
                img_idx=0,
                img_shape=img_shape,
                ori_shape=ori_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
            ),
            labels=labels,
        )


@TRANSFORMS.register_module(force=True)
class PackActionDetInputs(MMPackActionInputs):
    """Class to override PackActionInputs.

    Transfrom output dictionary from MMAction to ActionDetDataEntity.
    """

    def transform(self, results: dict) -> ActionDetDataEntity:
        """Transform function."""
        transformed = super().transform(results)
        image = tv_tensors.Image(transformed.get("inputs"))
        data_samples = transformed["data_samples"]

        ori_shape = results["original_shape"]
        img_shape = data_samples.img_shape
        pad_shape = data_samples.metainfo.get("pad_shape", img_shape)
        scale_factor = data_samples.metainfo.get("scale_factor", (1.0, 1.0))

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
        labels = results["__otx__"].labels

        return ActionDetDataEntity(
            image=image,
            img_info=ImageInfo(
                img_idx=0,
                img_shape=img_shape,
                ori_shape=ori_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
            ),
            bboxes=bboxes,
            labels=labels,
            proposals=proposals,
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
        transforms = [cls.get_builder().build(convert_conf_to_mmconfig_dict(cfg)) for cfg in config.transforms]

        cls._check_mandatory_transforms(
            transforms,
            mandatory_transforms={LoadVideo},
        )

        return transforms
