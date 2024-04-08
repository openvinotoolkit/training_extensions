# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Helper to support TorchVision data transform functions."""

from __future__ import annotations

from inspect import isclass
from typing import TYPE_CHECKING, Any

import numpy as np
import PIL.Image
import torch
import torchvision.transforms.v2 as tvt_v2
from datumaro.components.media import Video
from lightning.pytorch.cli import instantiate_class
from omegaconf import DictConfig
from torchvision import tv_tensors
from torchvision._utils import sequence_to_str
from torchvision.transforms.v2 import functional as F  # noqa: N812
from torchvision.transforms.v2._utils import get_bounding_boxes, query_size
from torchvision.ops.boxes import box_iou

from otx.core.data.entity.action_classification import ActionClsDataEntity
from otx.core.data.entity.base import Points

if TYPE_CHECKING:
    from torchvision.transforms.v2 import Compose

    from otx.core.config.data import SubsetConfig


def custom_query_size(flat_inputs: list[Any]) -> tuple[int, int]:  # noqa: D103
    sizes = {
        tuple(F.get_size(inpt))
        for inpt in flat_inputs
        if tvt_v2._utils.check_type(  # noqa: SLF001
            inpt,
            (
                F.is_pure_tensor,
                tv_tensors.Image,
                PIL.Image.Image,
                tv_tensors.Video,
                tv_tensors.Mask,
                tv_tensors.BoundingBoxes,
                Points,
            ),
        )
    }
    if not sizes:
        raise TypeError("No image, video, mask, bounding box, or point was found in the sample")  # noqa: EM101, TRY003
    elif len(sizes) > 1:  # noqa: RET506
        msg = f"Found multiple HxW dimensions in the sample: {sequence_to_str(sorted(sizes))}"
        raise ValueError(msg)
    h, w = sizes.pop()
    return h, w


tvt_v2._utils.query_size = custom_query_size  # noqa: SLF001


class PerturbBoundingBoxes(tvt_v2.Transform):
    """Perturb bounding boxes with random offset value."""

    def __init__(self, offset: int) -> None:
        super().__init__()
        self.offset = offset

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:  # noqa: ANN401
        output = self._perturb_bounding_boxes(inpt, self.offset)
        return tv_tensors.wrap(output, like=inpt)

    def _perturb_bounding_boxes(self, inpt: torch.Tensor, offset: int) -> torch.Tensor:
        mean = torch.zeros_like(inpt)
        repeated_size = torch.tensor(inpt.canvas_size).repeat(len(inpt), 2)
        std = torch.minimum(repeated_size * 0.1, torch.tensor(offset))
        noise = torch.normal(mean, std)
        return (inpt + noise).clamp(mean, repeated_size - 1)


class PadtoSquare(tvt_v2.Transform):
    """Pad skewed image to square with zero padding."""

    def _get_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        height, width = tvt_v2._utils.query_size(flat_inputs)  # noqa: SLF001
        max_dim = max(width, height)
        pad_w = max_dim - width
        pad_h = max_dim - height
        padding = (0, 0, pad_w, pad_h)
        return {"padding": padding}

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:  # noqa: ANN401
        return self._call_kernel(F.pad, inpt, padding=params["padding"], fill=0, padding_mode="constant")


class ResizetoLongestEdge(tvt_v2.Transform):
    """Resize image along with the longest edge."""

    def __init__(
        self,
        size: int,
        interpolation: F.InterpolationMode | int = F.InterpolationMode.BILINEAR,
        antialias: str | bool = "warn",
    ) -> None:
        super().__init__()

        self.size = size
        self.interpolation = F._geometry._check_interpolation(interpolation)  # noqa: SLF001
        self.antialias = antialias

    def _get_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        height, width = tvt_v2._utils.query_size(flat_inputs)  # noqa: SLF001
        target_size = self._get_preprocess_shape(height, width, self.size)
        return {"target_size": target_size}

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:  # noqa: ANN401
        return self._call_kernel(
            F.resize,
            inpt,
            params["target_size"],
            interpolation=self.interpolation,
            max_size=None,
            antialias=self.antialias,
        )

    def _get_preprocess_shape(self, oldh: int, oldw: int, long_side_length: int) -> tuple[int, int]:
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


class DecodeVideo(tvt_v2.Transform):
    """Sample video frames from original data video."""

    def __init__(
        self,
        test_mode: bool,
        clip_len: int = 8,
        frame_interval: int = 4,
        num_clips: int = 1,
        out_of_bound_opt: str = "loop",
    ) -> None:
        super().__init__()
        self.test_mode = test_mode
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.out_of_bound_opt = out_of_bound_opt
        self._transformed_types = [Video]

    def _transform(self, inpt: Video, params: dict) -> tv_tensors.Video:
        total_frames = self._get_total_frames(inpt)
        fps_scale_ratio = 1.0
        ori_clip_len = self._get_ori_clip_len(fps_scale_ratio)
        clip_offsets = self._sample_clips(total_frames, ori_clip_len)

        frame_inds = clip_offsets[:, None] + np.arange(self.clip_len)[None, :] * self.frame_interval
        frame_inds = np.concatenate(frame_inds)

        frame_inds = frame_inds.reshape((-1, self.clip_len))
        if self.out_of_bound_opt == "loop":
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == "repeat_last":
            safe_inds = frame_inds < total_frames
            unsafe_inds = 1 - safe_inds
            last_ind = np.max(safe_inds * frame_inds, axis=1)
            new_inds = safe_inds * frame_inds + (unsafe_inds.T * last_ind).T
            frame_inds = new_inds
        else:
            msg = "Illegal out_of_bound optio."
            raise ValueError(msg)

        start_index = 0
        frame_inds = np.concatenate(frame_inds) + start_index

        outputs = torch.stack([torch.tensor(inpt[idx].data) for idx in frame_inds], dim=0)
        outputs = outputs.permute(0, 3, 1, 2)
        outputs = tv_tensors.Video(outputs)
        inpt.close()

        return outputs

    @staticmethod
    def _get_total_frames(inpt: Video) -> int:
        length = 0
        for _ in inpt:
            length += 1
        return length

    def _get_train_clips(self, num_frames: int, ori_clip_len: float) -> np.array:
        """Get clip offsets in train mode.

        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.
            ori_clip_len (float): length of original sample clip.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

        if avg_interval > 0:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + np.random.default_rng().integers(avg_interval, size=self.num_clips)
        elif num_frames > max(self.num_clips, ori_clip_len):
            clip_offsets = np.sort(np.random.default_rng().integers(num_frames - ori_clip_len + 1, size=self.num_clips))
        elif avg_interval == 0:
            ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
            clip_offsets = np.around(np.arange(self.num_clips) * ratio)
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int32)

        return clip_offsets

    def _get_test_clips(self, num_frames: int, ori_clip_len: float) -> np.array:
        """Get clip offsets in test mode.

        If the total number of frames is
        not enough, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.
            ori_clip_len (float): length of original sample clip.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        max_offset = max(num_frames - ori_clip_len, 0)
        if self.num_clips > 1:
            num_segments = self.num_clips - 1
            # align test sample strategy with `PySlowFast` repo
            offset_between = max_offset / float(num_segments)
            clip_offsets = np.arange(self.num_clips) * offset_between
            clip_offsets = np.round(clip_offsets)
        else:
            clip_offsets = np.array([max_offset // 2])
        return clip_offsets

    def _sample_clips(self, num_frames: int, ori_clip_len: float) -> np.array:
        """Choose clip offsets for the video in a given mode.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames, ori_clip_len)
        else:
            clip_offsets = self._get_train_clips(num_frames, ori_clip_len)

        return clip_offsets

    def _get_ori_clip_len(self, fps_scale_ratio: float) -> float:
        """Calculate length of clip segment for different strategy.

        Args:
            fps_scale_ratio (float): Scale ratio to adjust fps.
        """
        if self.test_mode:
            ori_clip_len = (self.clip_len - 1) * self.frame_interval + 1
        else:
            ori_clip_len = self.clip_len * self.frame_interval

        return ori_clip_len


class PackVideo(tvt_v2.Transform):
    """Pack video for batch entity."""

    def forward(self, *inputs: ActionClsDataEntity) -> ActionClsDataEntity:
        """Replace ActionClsDataEntity's image to ActionClsDataEntity's video."""
        return inputs[0].wrap(image=inputs[0].video, video=[])


class MinIoURandomCrop(tvt_v2.RandomIoUCrop):
    """MinIoURandomCrop inherited from RandomIoUCrop to align with mmdet.transforms.MinIoURandomCrop.

    * Updated
        - change `ious.max()` to `ious.min()` at L121 because both MinIoURandomCrop and v2.RandomIoUCrop seems similar,
          but MinIoURandomCrop uses `overlaps.min()` to check if there is at least one box smaller than `min_iou` (https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L1432)
          and v2.RandomIoUCrop uses `ious.max()` to check if all boxes' IoUs are smaller than `min_jaccard_overlap` (https://github.com/pytorch/vision/blob/v0.16.1/torchvision/transforms/v2/_geometry.py#L1217).

        - `trials` in argument from 40 to 50 at L57.

    * Applied in other locations
        - box translation :
            mmdet : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L1454
            torchvision : https://github.com/pytorch/vision/blob/v0.16.1/torchvision/transforms/v2/functional/_geometry.py#L1386

        - clip border :
            mmdet : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L1456-L1457
            torchvision : https://github.com/pytorch/vision/blob/v0.16.1/torchvision/transforms/v2/functional/_geometry.py#L1389

        - except invalid bounding boxes :
            mmdet : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L1453
            torchvision : https://github.com/pytorch/vision/blob/v0.16.1/torchvision/transforms/v2/_geometry.py#L1232-L1234
                + SanitizeBoundingBoxes

    Args:
        min_scale (float, optional): Minimum factors to scale the input size.
        max_scale (float, optional): Maximum factors to scale the input size.
        min_aspect_ratio (float, optional): Minimum aspect ratio for the cropped image or video.
        max_aspect_ratio (float, optional): Maximum aspect ratio for the cropped image or video.
        sampler_options (list of float, optional): List of minimal IoU (Jaccard) overlap between all the boxes and
            a cropped image or video. Default, ``None`` which corresponds to ``[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]``
        trials (int, optional): Number of trials to find a crop for a given value of minimal IoU (Jaccard) overlap.
            Default, 50.
    """

    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1.0,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2.0,
        sampler_options: list[float] | None = None,
        trials: int = 50,  # 40 -> 50
    ):
        super().__init__(
            min_scale=min_scale,
            max_scale=max_scale,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
            sampler_options=sampler_options,
            trials=trials,
        )

    def _get_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        orig_h, orig_w = query_size(flat_inputs)
        bboxes = get_bounding_boxes(flat_inputs)

        while True:
            # https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L1407-L1410
            # sample an option
            idx = int(torch.randint(low=0, high=len(self.options), size=(1,)))
            min_jaccard_overlap = self.options[idx]
            if min_jaccard_overlap >= 1.0:  # a value larger than 1 encodes the leave as-is option
                return {}

            for _ in range(self.trials):
                # https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L1414-L1419
                # check the aspect ratio limitations
                r = self.min_scale + (self.max_scale - self.min_scale) * torch.rand(2)
                new_w = int(orig_w * r[0])
                new_h = int(orig_h * r[1])
                aspect_ratio = new_w / new_h
                if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                    continue

                # https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L1421-L1428
                # check for 0 area crops
                r = torch.rand(2)
                left = int((orig_w - new_w) * r[0])
                top = int((orig_h - new_h) * r[1])
                right = left + new_w
                bottom = top + new_h
                if left == right or top == bottom:
                    continue

                # https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L1439-L1445
                # https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L1452-L1453
                # check for any valid boxes with centers within the crop area
                xyxy_bboxes = F.convert_bounding_box_format(
                    bboxes.as_subclass(torch.Tensor),
                    bboxes.format,
                    tv_tensors.BoundingBoxFormat.XYXY,
                )
                # cx = 0.5 * (xyxy_bboxes[..., 0] + xyxy_bboxes[..., 2])
                # cy = 0.5 * (xyxy_bboxes[..., 1] + xyxy_bboxes[..., 3])
                # is_within_crop_area = (left < cx) & (cx < right) & (top < cy) & (cy < bottom)
                # if not is_within_crop_area.any():
                #     continue

                # # https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L1429-L1433
                # xyxy_bboxes = xyxy_bboxes[is_within_crop_area]
                ious = box_iou(
                    xyxy_bboxes,
                    torch.tensor([[left, top, right, bottom]], dtype=xyxy_bboxes.dtype, device=xyxy_bboxes.device),
                )
                if ious.min() < min_jaccard_overlap:  # max -> min
                    continue

                cx = 0.5 * (xyxy_bboxes[..., 0] + xyxy_bboxes[..., 2])
                cy = 0.5 * (xyxy_bboxes[..., 1] + xyxy_bboxes[..., 3])
                is_within_crop_area = (left < cx) & (cx < right) & (top < cy) & (cy < bottom)
                if not is_within_crop_area.any():
                    continue

                return {
                    "top": top,
                    "left": left,
                    "height": new_h,
                    "width": new_w,
                    "is_within_crop_area": is_within_crop_area,
                }


tvt_v2.PerturbBoundingBoxes = PerturbBoundingBoxes
tvt_v2.PadtoSquare = PadtoSquare
tvt_v2.ResizetoLongestEdge = ResizetoLongestEdge


class TorchVisionTransformLib:
    """Helper to support TorchVision transforms (only V2) in OTX."""

    @classmethod
    def list_available_transforms(cls) -> list[type[tvt_v2.Transform]]:
        """List available TorchVision transform (only V2) classes."""
        return [
            obj
            for name in dir(tvt_v2)
            if (obj := getattr(tvt_v2, name)) and isclass(obj) and issubclass(obj, tvt_v2.Transform)
        ]

    @classmethod
    def generate(cls, config: SubsetConfig) -> Compose:
        """Generate TorchVision transforms from the configuration."""
        if isinstance(config.transforms, tvt_v2.Compose):
            return config.transforms

        transforms = []
        for cfg_transform in config.transforms:
            transform = cls._dispatch_transform(cfg_transform)
            transforms.append(transform)

        return tvt_v2.Compose(transforms)

    @classmethod
    def _dispatch_transform(cls, cfg_transform: DictConfig | dict | tvt_v2.Transform) -> tvt_v2.Transform:
        if isinstance(cfg_transform, (DictConfig, dict)):
            transform = instantiate_class(args=(), init=cfg_transform)

        elif isinstance(cfg_transform, tvt_v2.Transform):
            transform = cfg_transform
        else:
            msg = (
                "TorchVisionTransformLib accepts only three types "
                "for config.transforms: DictConfig | dict | tvt_v2.Transform. "
                f"However, its type is {type(cfg_transform)}."
            )
            raise TypeError(msg)

        return transform
