# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Helper to support TorchVision data transform functions."""

from __future__ import annotations

import copy
import random
from inspect import isclass
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import PIL.Image
import torch
import torchvision.transforms.v2 as tvt_v2
from datumaro.components.media import Video
from lightning.pytorch.cli import instantiate_class
from omegaconf import DictConfig
from torchvision import tv_tensors
from torchvision._utils import sequence_to_str
from torchvision.ops.boxes import box_iou
from torchvision.transforms.v2 import functional as F  # noqa: N812
from torchvision.transforms.v2._utils import get_bounding_boxes, query_size
from torchvision.transforms.v2.functional._color import _rgb_to_hsv, _hsv_to_rgb

from otx.core.data.entity.action_classification import ActionClsDataEntity
from otx.core.data.entity.base import Points, _resize_image_info
from otx.core.data.transform_libs.utils import (cache_randomness, clip_bboxes,
                                                rescale_bboxes,
                                                translate_bboxes, is_inside_bboxes, flip_bboxes)

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

    def __init__(self, pad_val: int = 0) -> None:
        super().__init__()
        self.pad_val = pad_val

    def _get_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        height, width = tvt_v2._utils.query_size(flat_inputs)  # noqa: SLF001
        max_dim = max(width, height)
        pad_w = max_dim - width
        pad_h = max_dim - height
        padding = (0, 0, pad_w, pad_h)
        return {"padding": padding}

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:  # noqa: ANN401
        return self._call_kernel(F.pad, inpt, padding=params["padding"], fill=self.pad_val, padding_mode="constant")


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


class CachedMosaic(tvt_v2.Transform):
    """CachedMosaic converted from mmdet.datasets.transforms.CachedMosaic.
    
    Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L3342-L3573

    TODO : update masks for instance segmentation
    TODO : optimize logic to torcivision pipeline

    Args:
        img_scale (Sequence[int]): Image size before mosaic pipeline of single
            image. The shape order should be (width, height).
            Defaults to (640, 640).
        center_ratio_range (Sequence[float]): Center ratio range of mosaic
            output. Defaults to (0.5, 1.5).
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        pad_val (int): Pad value. Defaults to 114.
        prob (float): Probability of applying this transformation.
            Defaults to 1.0.
        max_cached_images (int): The maximum length of the cache. The larger
            the cache, the stronger the randomness of this transform. As a
            rule of thumb, providing 10 caches for each image suffices for
            randomness. Defaults to 40.
        random_pop (bool): Whether to randomly pop a result from the cache
            when the cache is full. If set to False, use FIFO popping method.
            Defaults to True.
    """
    def __init__(
        self,
        img_scale: tuple[int, int] | list[int] = (640, 640), # (H, W)
        center_ratio_range: tuple[float, float] = (0.5, 1.5),
        bbox_clip_border: bool = True,
        pad_val: float = 114.0,
        prob: float = 1.0,
        max_cached_images: int = 40,
        random_pop: bool = True,
    ) -> None:

        super().__init__()

        assert isinstance(img_scale, (tuple, list))
        assert 0 <= prob <= 1.0, "The probability should be in range [0,1]. " \
                                 f"got {prob}."

        self.img_scale = img_scale # HW
        self.center_ratio_range = center_ratio_range
        self.bbox_clip_border = bbox_clip_border
        self.pad_val = pad_val
        self.prob = prob

        self.results_cache = []
        self.random_pop = random_pop
        assert max_cached_images >= 4, "The length of cache must >= 4, " \
                                       f"but got {max_cached_images}."
        self.max_cached_images = max_cached_images

        self.cnt_cached_images = 0

    @cache_randomness
    def get_indexes(self, cache: list) -> list:
        """Call function to collect indexes.

        Args:
            cache (list): The results cache.

        Returns:
            list: indexes.
        """

        indexes = [random.randint(0, len(cache) - 1) for _ in range(3)]
        return indexes

    def forward(self, *inputs: Any) -> Any:
        assert len(inputs) == 1, "[tmp] Multiple entity is not supported yet."
        inputs = inputs[0]

        self.results_cache.append(copy.deepcopy(inputs))
        if len(self.results_cache) > self.max_cached_images:
            if self.random_pop:
                index = random.randint(0, len(self.results_cache) - 1)
            else:
                index = 0
            self.results_cache.pop(index)

        if len(self.results_cache) <= 4:
            return inputs

        if random.uniform(0, 1) > self.prob:
            return inputs

        indices = self.get_indexes(self.results_cache)
        mix_results = [copy.deepcopy(self.results_cache[i]) for i in indices]

        # TODO: refactor mosaic to reuse these code.
        mosaic_bboxes = []
        mosaic_bboxes_labels = []

        if inputs.image.ndim == 3:
            mosaic_img = torch.full(
                (3, int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)), # CHW
                self.pad_val,
                dtype=inputs.image.dtype)
        else:
            mosaic_img = torch.full(
                (1, int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                self.pad_val,
                dtype=inputs.image.dtype)

        # mosaic center x, y
        center_x = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_y = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_position = (center_x, center_y)

        loc_strs = ("top_left", "top_right", "bottom_left", "bottom_right")
        for i, loc in enumerate(loc_strs):
            if loc == "top_left":
                results_patch = copy.deepcopy(inputs)
            else:
                results_patch = copy.deepcopy(mix_results[i - 1])

            img_i = results_patch.image
            h_i, w_i = img_i.shape[-2:]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[0] / h_i,
                                self.img_scale[1] / w_i)
            img_i = F.resize(img_i, [int(h_i * scale_ratio_i), int(w_i * scale_ratio_i)])

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[-2:][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[..., y1_p:y2_p, x1_p:x2_p] = img_i[..., y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_bboxes_i = results_patch.bboxes
            gt_bboxes_labels_i = results_patch.labels

            padw = x1_p - x1_c
            padh = y1_p - y1_c
            # TODO (sungchul): use tv dispatch (private) or below
            gt_bboxes_i = rescale_bboxes(gt_bboxes_i, [scale_ratio_i, scale_ratio_i])
            gt_bboxes_i = translate_bboxes(gt_bboxes_i, [padw, padh])
            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_bboxes_labels.append(gt_bboxes_labels_i)

        mosaic_bboxes = torch.cat(mosaic_bboxes, dim=0)
        mosaic_bboxes_labels = torch.cat(mosaic_bboxes_labels, dim=0)

        if self.bbox_clip_border:
            mosaic_bboxes = clip_bboxes(mosaic_bboxes, [2 * self.img_scale[0], 2 * self.img_scale[1]])

        # remove outside bboxes
        inside_inds = is_inside_bboxes(mosaic_bboxes, [2 * self.img_scale[0], 2 * self.img_scale[1]])
        mosaic_bboxes = mosaic_bboxes[inside_inds]
        mosaic_bboxes_labels = mosaic_bboxes_labels[inside_inds]

        inputs.image = tv_tensors.Image(mosaic_img)
        inputs.img_info = _resize_image_info(inputs.img_info, mosaic_img.shape[-2:])
        inputs.bboxes = tv_tensors.BoundingBoxes(mosaic_bboxes, format="XYXY", canvas_size=mosaic_img.shape[-2:])
        inputs.labels = mosaic_bboxes_labels
        return inputs

    def _mosaic_combine(self, loc: str, center_position_xy: Sequence[float], img_shape_wh: Sequence[int]) -> tuple[tuple[int], tuple[int]]:
        """Calculate global coordinate of mosaic image and local coordinate of
        cropped sub-image.

        Args:
            loc (str): Index for the sub-image, loc in ("top_left",
              "top_right", "bottom_left", "bottom_right").
            center_position_xy (Sequence[float]): Mixing center for 4 images,
                (x, y).
            img_shape_wh (Sequence[int]): Width and height of sub-image

        Returns:
            tuple[tuple[float]]: Corresponding coordinate of pasting and
                cropping
                - paste_coord (tuple): paste corner coordinate in mosaic image.
                - crop_coord (tuple): crop corner coordinate in mosaic image.
        """
        assert loc in ("top_left", "top_right", "bottom_left", "bottom_right")
        if loc == "top_left":
            # index0 to top left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             center_position_xy[0], \
                             center_position_xy[1]
            crop_coord = img_shape_wh[0] - (x2 - x1), img_shape_wh[1] - (
                y2 - y1), img_shape_wh[0], img_shape_wh[1]

        elif loc == "top_right":
            # index1 to top right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[0] * 2), \
                             center_position_xy[1]
            crop_coord = 0, img_shape_wh[1] - (y2 - y1), min(
                img_shape_wh[0], x2 - x1), img_shape_wh[1]

        elif loc == "bottom_left":
            # index2 to bottom left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             center_position_xy[1], \
                             center_position_xy[0], \
                             min(self.img_scale[1] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = img_shape_wh[0] - (x2 - x1), 0, img_shape_wh[0], min(
                y2 - y1, img_shape_wh[1])

        else:
            # index3 to bottom right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             center_position_xy[1], \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[0] * 2), \
                             min(self.img_scale[1] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = 0, 0, min(img_shape_wh[0],
                                   x2 - x1), min(y2 - y1, img_shape_wh[1])

        paste_coord = x1, y1, x2, y2
        return paste_coord, crop_coord

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'center_ratio_range={self.center_ratio_range}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'max_cached_images={self.max_cached_images}, '
        repr_str += f'random_pop={self.random_pop})'
        return repr_str


class CachedMixUp(tvt_v2.Transform):
    """CachedMixup converted from mmdet.datasets.transforms.CachedMixup.
    
    Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L3577-L3854

    TODO : update masks for instance segmentation
    TODO : optimize logic to torcivision pipeline

    Args:
        img_scale (Sequence[int]): Image output size after mixup pipeline.
            The shape order should be (width, height). Defaults to (640, 640).
        ratio_range (Sequence[float]): Scale ratio of mixup image.
            Defaults to (0.5, 1.5).
        flip_ratio (float): Horizontal flip ratio of mixup image.
            Defaults to 0.5.
        pad_val (int): Pad value. Defaults to 114.
        max_iters (int): The maximum number of iterations. If the number of
            iterations is greater than `max_iters`, but gt_bbox is still
            empty, then the iteration is terminated. Defaults to 15.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        max_cached_images (int): The maximum length of the cache. The larger
            the cache, the stronger the randomness of this transform. As a
            rule of thumb, providing 10 caches for each image suffices for
            randomness. Defaults to 20.
        random_pop (bool): Whether to randomly pop a result from the cache
            when the cache is full. If set to False, use FIFO popping method.
            Defaults to True.
        prob (float): Probability of applying this transformation.
            Defaults to 1.0.
    """

    def __init__(self,
                 img_scale: tuple[int, int] | list[int] = (640, 640),
                 ratio_range: tuple[float, float] = (0.5, 1.5),
                 flip_ratio: float = 0.5,
                 pad_val: float = 114.0,
                 max_iters: int = 15,
                 bbox_clip_border: bool = True,
                 max_cached_images: int = 20,
                 random_pop: bool = True,
                 prob: float = 1.0) -> None:
        super().__init__()

        assert isinstance(img_scale, (tuple, list))
        assert max_cached_images >= 2, 'The length of cache must >= 2, ' \
                                       f'but got {max_cached_images}.'
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. ' \
                                 f'got {prob}.'
        self.dynamic_scale = img_scale # HW
        self.ratio_range = ratio_range
        self.flip_ratio = flip_ratio
        self.pad_val = pad_val
        self.max_iters = max_iters
        self.bbox_clip_border = bbox_clip_border
        self.results_cache = []

        self.max_cached_images = max_cached_images
        self.random_pop = random_pop
        self.prob = prob

    @cache_randomness
    def get_indexes(self, cache: list) -> int:
        """Call function to collect indexes.

        Args:
            cache (list): The result cache.

        Returns:
            int: index.
        """

        for i in range(self.max_iters):
            index = random.randint(0, len(cache) - 1)
            gt_bboxes_i = cache[index].bboxes
            if len(gt_bboxes_i) != 0:
                break
        return index

    def forward(self, *inputs: Any) -> Any:
        """MixUp transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        # cache and pop images
        assert len(inputs) == 1, "[tmp] Multiple entity is not supported yet."
        inputs = inputs[0]

        self.results_cache.append(copy.deepcopy(inputs))
        if len(self.results_cache) > self.max_cached_images:
            if self.random_pop:
                index = random.randint(0, len(self.results_cache) - 1)
            else:
                index = 0
            self.results_cache.pop(index)

        if len(self.results_cache) <= 1:
            return inputs

        if random.uniform(0, 1) > self.prob:
            return inputs

        index = self.get_indexes(self.results_cache)
        retrieve_results = copy.deepcopy(self.results_cache[index])

        # TODO: refactor mixup to reuse these code.
        if retrieve_results.bboxes.shape[0] == 0:
            # empty bbox
            return inputs

        retrieve_img = retrieve_results.image

        jit_factor = random.uniform(*self.ratio_range)
        is_flip = random.uniform(0, 1) > self.flip_ratio

        out_img = torch.ones(
            (3, self.dynamic_scale[0], self.dynamic_scale[1]),
            dtype=retrieve_img.dtype) * torch.tensor(self.pad_val).unsqueeze(-1).unsqueeze(-1)

        # 1. keep_ratio resize
        scale_ratio = min(self.dynamic_scale[0] / retrieve_img.shape[-2],
                          self.dynamic_scale[1] / retrieve_img.shape[-1])
        retrieve_img = F.resize(
            retrieve_img, [int(retrieve_img.shape[-2] * scale_ratio),
                           int(retrieve_img.shape[-1] * scale_ratio)])

        # 2. paste
        out_img[:, :retrieve_img.shape[-2], :retrieve_img.shape[-1]] = retrieve_img

        # 3. scale jit
        scale_ratio *= jit_factor
        out_img = F.resize(out_img, (int(out_img.shape[-2] * jit_factor), int(out_img.shape[-1] * jit_factor)))

        # 4. flip
        if is_flip:
            out_img = F.horizontal_flip_image(out_img)

        # 5. random crop
        ori_img = inputs.image
        origin_h, origin_w = out_img.shape[-2:]
        target_h, target_w = ori_img.shape[-2:]
        padded_img = torch.ones((3, max(origin_h, target_h), max(origin_w, target_w)), dtype=torch.uint8) * torch.tensor(self.pad_val).unsqueeze(-1).unsqueeze(-1)
        padded_img[..., :origin_h, :origin_w] = out_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[-2] > target_h:
            y_offset = random.randint(0, padded_img.shape[-2] - target_h)
        if padded_img.shape[-1] > target_w:
            x_offset = random.randint(0, padded_img.shape[-1] - target_w)
        padded_cropped_img = padded_img[..., y_offset:y_offset + target_h, x_offset:x_offset + target_w]

        # 6. adjust bbox
        retrieve_gt_bboxes = retrieve_results.bboxes
        retrieve_gt_bboxes = rescale_bboxes(retrieve_gt_bboxes, [scale_ratio, scale_ratio])

        if self.bbox_clip_border:
            retrieve_gt_bboxes = clip_bboxes(retrieve_gt_bboxes, [origin_h, origin_w])

        if is_flip:
            retrieve_gt_bboxes = flip_bboxes(retrieve_gt_bboxes, [origin_h, origin_w], direction="horizontal")

        # 7. filter
        cp_retrieve_gt_bboxes = retrieve_gt_bboxes.clone()
        cp_retrieve_gt_bboxes = translate_bboxes(cp_retrieve_gt_bboxes, [-x_offset, -y_offset])

        if self.bbox_clip_border:
            cp_retrieve_gt_bboxes = clip_bboxes(cp_retrieve_gt_bboxes, [target_h, target_w])

        # 8. mix up
        ori_img = ori_img.to(torch.float32)
        mixup_img = 0.5 * ori_img + 0.5 * padded_cropped_img.to(torch.float32)

        retrieve_gt_bboxes_labels = retrieve_results.labels

        mixup_gt_bboxes = torch.cat(
            (inputs.bboxes, cp_retrieve_gt_bboxes), dim=0)
        mixup_gt_bboxes_labels = torch.cat(
            (inputs.labels, retrieve_gt_bboxes_labels), dim=0)

        # remove outside bbox
        inside_inds = is_inside_bboxes(mixup_gt_bboxes, [target_h, target_w])
        mixup_gt_bboxes = mixup_gt_bboxes[inside_inds]
        mixup_gt_bboxes_labels = mixup_gt_bboxes_labels[inside_inds]

        inputs.image = tv_tensors.Image(mixup_img.to(torch.uint8))
        inputs.img_info = _resize_image_info(inputs.img_info, mixup_img.shape[-2:])
        inputs.bboxes = tv_tensors.BoundingBoxes(mixup_gt_bboxes, format="XYXY", canvas_size=mixup_img.shape[-2:])
        inputs.labels = mixup_gt_bboxes_labels
        return inputs

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(dynamic_scale={self.dynamic_scale}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'flip_ratio={self.flip_ratio}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'max_iters={self.max_iters}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border}, '
        repr_str += f'max_cached_images={self.max_cached_images}, '
        repr_str += f'random_pop={self.random_pop}, '
        repr_str += f'prob={self.prob})'
        return repr_str


class YOLOXHSVRandomAug(tvt_v2.Transform):
    """YOLOXHSVRandomAug converted from mmdet.datasets.transforms.YOLOXHSVRandomAug.
    
    Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L2905-L2961

    TODO : update masks for instance segmentation
    TODO : optimize logic to torcivision pipeline

    Args:
        hue_delta (int): delta of hue. Defaults to 5.
        saturation_delta (int): delta of saturation. Defaults to 30.
        value_delta (int): delat of value. Defaults to 30.
    """

    def __init__(self,
                 hue_delta: int = 5,
                 saturation_delta: int = 30,
                 value_delta: int = 30) -> None:
        super().__init__()
        
        self.hue_delta = hue_delta
        self.saturation_delta = saturation_delta
        self.value_delta = value_delta

    @cache_randomness
    def _get_hsv_gains(self):
        hsv_gains = torch.FloatTensor(3).uniform_(-1, 1) * torch.tensor([
            self.hue_delta, self.saturation_delta, self.value_delta
        ])
        # random selection of h, s, v
        hsv_gains *= torch.randint(0, 2, (3,))
        # prevent overflow
        hsv_gains = hsv_gains.to(torch.int16)
        return hsv_gains

    def forward(self, *inputs: Any) -> Any:
        assert len(inputs) == 1, "[tmp] Multiple entity is not supported yet."
        inputs = inputs[0]

        img = inputs.image
        hsv_gains = self._get_hsv_gains()
        img_hsv = _rgb_to_hsv(img).to(torch.int16)

        img_hsv[..., 0] = (img_hsv[..., 0] + hsv_gains[0]) % 180
        img_hsv[..., 1] = torch.clip(img_hsv[..., 1] + hsv_gains[1], 0, 255)
        img_hsv[..., 2] = torch.clip(img_hsv[..., 2] + hsv_gains[2], 0, 255)
        
        # TODO (sungchul): check if _hsv_to_rgb only supports fp, not uint8
        inputs.image = tv_tensors.Image((_hsv_to_rgb(img_hsv / 255) * 255).to(img.dtype))
        return inputs

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(hue_delta={self.hue_delta}, '
        repr_str += f'saturation_delta={self.saturation_delta}, '
        repr_str += f'value_delta={self.value_delta})'
        return repr_str
    

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
