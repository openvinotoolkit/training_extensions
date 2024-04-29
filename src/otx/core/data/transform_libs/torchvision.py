# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Helper to support TorchVision data transform functions."""

from __future__ import annotations

import copy
import math
from inspect import isclass
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Sequence

import cv2
import numpy as np
import PIL.Image
import torch
import torchvision.transforms.v2 as tvt_v2
from datumaro.components.media import Video
from lightning.pytorch.cli import instantiate_class
from numpy import random
from omegaconf import DictConfig
from torchvision import tv_tensors
from torchvision._utils import sequence_to_str
from torchvision.transforms.v2 import functional as F  # noqa: N812

from otx.core.data.entity.action_classification import ActionClsDataEntity
from otx.core.data.entity.base import (
    Points,
    _crop_image_info,
    _pad_image_info,
    _resize_image_info,
    _resized_crop_image_info,
)
from otx.core.data.transform_libs.utils import (
    cache_randomness,
    centers_bboxes,
    clip_bboxes,
    flip_bboxes,
    flip_image,
    is_inside_bboxes,
    overlap_bboxes,
    project_bboxes,
    rescale_bboxes,
    rescale_size,
    scale_size,
    to_np_image,
    translate_bboxes,
)

if TYPE_CHECKING:
    from torchvision.transforms.v2 import Compose

    from otx.core.config.data import SubsetConfig
    from otx.core.data.entity.base import T_OTXDataEntity
    from otx.core.data.entity.detection import DetDataEntity


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


class MinIoURandomCrop(tvt_v2.Transform):
    """Implementation of mmdet.datasets.transforms.MinIoURandomCrop with torchvision format.

    Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L1338-L1490

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
        min_ious: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size: float = 0.3,
        bbox_clip_border: bool = True,
    ) -> None:
        super().__init__()
        self.min_ious = min_ious
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size
        self.bbox_clip_border = bbox_clip_border

    @cache_randomness
    def _random_mode(self) -> int | float:
        return random.choice(self.sample_mode)

    def forward(self, *_inputs: DetDataEntity) -> DetDataEntity:
        """Forward for MinIoURandomCrop."""
        assert len(_inputs) == 1, "[tmp] Multiple entity is not supported yet."  # noqa: S101
        inputs = _inputs[0]

        img = to_np_image(inputs.image)
        boxes = inputs.bboxes
        h, w, c = img.shape
        while True:
            mode = self._random_mode()
            self.mode = mode
            if mode == 1:
                return inputs

            min_iou = self.mode
            for _ in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array((int(left), int(top), int(left + new_w), int(top + new_h)))
                # Line or point crop is not allowed
                if patch[2] == patch[0] or patch[3] == patch[1]:
                    continue
                overlaps = (
                    overlap_bboxes(torch.as_tensor(patch.reshape(-1, 4).astype(np.float32)), boxes).numpy().reshape(-1)
                )
                if len(overlaps) > 0 and overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                # only adjust boxes and instance masks when the gt is not empty
                if len(overlaps) > 0:
                    # adjust boxes
                    def is_center_of_bboxes_in_patch(boxes: torch.Tensor, patch: np.ndarray) -> np.ndarray:
                        centers = centers_bboxes(boxes).numpy()
                        return (
                            (centers[:, 0] > patch[0])
                            * (centers[:, 1] > patch[1])
                            * (centers[:, 0] < patch[2])
                            * (centers[:, 1] < patch[3])
                        )

                    mask = is_center_of_bboxes_in_patch(boxes, patch)
                    if not mask.any():
                        continue
                    if (bboxes := getattr(inputs, "bboxes", None)) is not None:
                        mask = is_center_of_bboxes_in_patch(bboxes, patch)
                        bboxes = bboxes[mask]
                        bboxes = translate_bboxes(bboxes, (-patch[0], -patch[1]))
                        if self.bbox_clip_border:
                            bboxes = clip_bboxes(bboxes, (patch[3] - patch[1], patch[2] - patch[0]))
                        inputs.bboxes = tv_tensors.BoundingBoxes(
                            bboxes,
                            format="XYXY",
                            canvas_size=(patch[3] - patch[1], patch[2] - patch[0]),
                        )

                        # labels
                        if (labels := getattr(inputs, "labels", None)) is not None:
                            inputs.labels = labels[mask]

                # adjust the img no matter whether the gt is empty before crop
                img = img[patch[1] : patch[3], patch[0] : patch[2]]
                inputs.image = img
                inputs.img_info = _crop_image_info(inputs.img_info, *img.shape[:2])
                return inputs

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(min_ious={self.min_ious}, "
        repr_str += f"min_crop_size={self.min_crop_size}, "
        repr_str += f"bbox_clip_border={self.bbox_clip_border})"
        return repr_str


class Resize(tvt_v2.Transform):
    """Implementation of mmdet.datasets.transforms.Resize with torchvision format.

    Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L135-L246

    TODO : update masks for instance segmentation
    TODO : optimize logic to torcivision pipeline

    Args:
        scale (int or tuple): Images scales for resizing with (height, width). Defaults to None
        scale_factor (float or tuple[float]): Scale factors for resizing with (height, width).
            Defaults to None.
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Defaults to False.
        clip_object_border (bool): Whether to clip the objects
            outside the border of the image. In some dataset like MOT17, the gt
            bboxes are allowed to cross the border of images. Therefore, we
            don't need to clip the gt bboxes in these cases. Defaults to True.
        interpolation (str): Interpolation method for cv2. Defaults to 'bilinear'.
    """

    cv2_interp_codes: ClassVar = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4,
    }

    def __init__(
        self,
        scale: int | tuple[int, int] | None = None,
        scale_factor: float | tuple[float, float] | None = None,
        keep_ratio: bool = False,
        clip_object_border: bool = True,
        interpolation: str = "bilinear",
        transform_bbox: bool = True,
    ) -> None:
        super().__init__()

        assert scale is not None or scale_factor is not None, "`scale` and`scale_factor` can not both be `None`"  # noqa: S101

        if scale is None:
            self.scale = None
        elif isinstance(scale, int):
            self.scale = (scale, scale)
        else:
            self.scale = tuple(scale)  # type: ignore[assignment]

        self.transform_bbox = transform_bbox
        self.interpolation = interpolation
        self.keep_ratio = keep_ratio
        self.clip_object_border = clip_object_border
        if scale_factor is None:
            self.scale_factor = None
        elif isinstance(scale_factor, float):
            self.scale_factor = (scale_factor, scale_factor)
        elif isinstance(scale_factor, tuple) and len(scale_factor) == 2:
            self.scale_factor = scale_factor
        else:
            msg = f"expect scale_factor is float or Tuple(float), butget {type(scale_factor)}"
            raise TypeError(msg)

    def _resize_img(self, inputs: T_OTXDataEntity) -> tuple[T_OTXDataEntity, tuple[float, float] | None]:
        """Resize images with inputs.img_info.img_shape."""
        scale_factor: tuple[float, float] | None = getattr(inputs.img_info, "scale_factor", None)
        if (img := getattr(inputs, "image", None)) is not None:
            img = to_np_image(img)

            img_shape = img.shape[:2]
            scale: tuple[int, int] = self.scale or scale_size(img_shape[::-1], self.scale_factor)  # type: ignore[arg-type]

            if self.keep_ratio:
                scale = rescale_size(img_shape[::-1], scale)  # type: ignore[assignment]

            img = cv2.resize(img, scale, interpolation=self.cv2_interp_codes[self.interpolation])

            inputs.image = img
            inputs.img_info = _resize_image_info(inputs.img_info, img.shape[:2])

            scale_factor = (scale[0] / img_shape[1], scale[1] / img_shape[0])  # TODO (sungchul): ticket no. 138831
        return inputs, scale_factor

    def _resize_bboxes(self, inputs: DetDataEntity, scale_factor: tuple[float, float]) -> DetDataEntity:
        """Resize bounding boxes with inputs.img_info.scale_factor."""
        if (bboxes := getattr(inputs, "bboxes", None)) is not None:
            bboxes = rescale_bboxes(bboxes, scale_factor)  # TODO (sungchul): ticket no. 138831
            if self.clip_object_border:
                bboxes = clip_bboxes(bboxes, inputs.img_info.img_shape)
            inputs.bboxes = tv_tensors.BoundingBoxes(bboxes, format="XYXY", canvas_size=inputs.img_info.img_shape)
        return inputs

    def forward(self, *_inputs: T_OTXDataEntity) -> T_OTXDataEntity:
        """Transform function to resize images and bounding boxes.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'scale', 'scale_factor', 'height', 'width', and 'keep_ratio' keys
            are updated in result dict.
        """
        assert len(_inputs) == 1, "[tmp] Multiple entity is not supported yet."  # noqa: S101
        inputs = _inputs[0]

        inputs, scale_factor = self._resize_img(inputs)
        if self.transform_bbox:
            inputs = self._resize_bboxes(inputs, scale_factor)  # type: ignore[arg-type, assignment]

        return inputs

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(scale={self.scale}, "
        repr_str += f"scale_factor={self.scale_factor}, "
        repr_str += f"keep_ratio={self.keep_ratio}, "
        repr_str += f"clip_object_border={self.clip_object_border}), "
        repr_str += f"interpolation={self.interpolation})"
        return repr_str


class RandomResizedCrop(tvt_v2.Transform):
    """Crop the given image to random scale and aspect ratio.

    This class implements mmpretrain.datasets.transforms.RandomResizedCrop reimplemented as torchvision.transform.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a
    random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio
    is made. This crop is finally resized to given size.

    Args:
        scale (sequence | int): Desired output scale of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        crop_ratio_range (tuple): Range of the random size of the cropped
            image compared to the original image. Defaults to (0.08, 1.0).
        aspect_ratio_range (tuple): Range of the random aspect ratio of the
            cropped image compared to the original image.
            Defaults to (3. / 4., 4. / 3.).
        max_attempts (int): Maximum number of attempts before falling back to
            Central Crop. Defaults to 10.
        interpolation (str): Interpolation method, accepted values are
            'nearest', 'bilinear', 'bicubic', 'area', 'lanczos'. Defaults to
            'bilinear'.
        backend (str): The image resize backend type, accepted values are
            'cv2' and 'pillow'. Defaults to 'cv2'.
    """

    cv2_interp_codes: ClassVar = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4,
    }

    def __init__(
        self,
        scale: Sequence | int,
        crop_ratio_range: tuple[float, float] = (0.08, 1.0),
        aspect_ratio_range: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        max_attempts: int = 10,
        interpolation: str = "bilinear",
        backend: str = "cv2",
    ) -> None:
        super().__init__()
        if isinstance(scale, Sequence):
            assert len(scale) == 2  # noqa: S101
            assert scale[0] > 0  # noqa: S101
            assert scale[1] > 0  # noqa: S101
            self.scale = scale
        else:
            assert scale > 0  # noqa: S101
            self.scale = (scale, scale)
        if (crop_ratio_range[0] > crop_ratio_range[1]) or (aspect_ratio_range[0] > aspect_ratio_range[1]):
            msg = (
                "range should be of kind (min, max). "
                f"But received crop_ratio_range {crop_ratio_range} "
                f"and aspect_ratio_range {aspect_ratio_range}."
            )
            raise ValueError(msg)
        assert isinstance(max_attempts, int)  # noqa: S101
        assert max_attempts >= 0, "max_attempts mush be int and no less than 0."  # noqa: S101
        assert interpolation in (  # noqa: S101
            "nearest",
            "bilinear",
            "bicubic",
            "area",
            "lanczos",
        )

        self.crop_ratio_range = crop_ratio_range
        self.aspect_ratio_range = aspect_ratio_range
        self.max_attempts = max_attempts
        self.interpolation = interpolation
        self.backend = backend

    @cache_randomness
    def rand_crop_params(self, img: np.ndarray) -> tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (ndarray): Image to be cropped.

        Returns:
            tuple: Params (offset_h, offset_w, target_h, target_w) to be
                passed to `crop` for a random sized crop.
        """
        h, w = img.shape[:2]
        area = h * w

        for _ in range(self.max_attempts):
            target_area = np.random.uniform(*self.crop_ratio_range) * area
            log_ratio = (math.log(self.aspect_ratio_range[0]), math.log(self.aspect_ratio_range[1]))
            aspect_ratio = math.exp(np.random.uniform(*log_ratio))
            target_w = int(round(math.sqrt(target_area * aspect_ratio)))
            target_h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < target_w <= w and 0 < target_h <= h:
                offset_h = np.random.randint(0, h - target_h + 1)
                offset_w = np.random.randint(0, w - target_w + 1)

                return offset_h, offset_w, target_h, target_w

        # Fallback to central crop
        in_ratio = float(w) / float(h)
        if in_ratio < min(self.aspect_ratio_range):
            target_w = w
            target_h = int(round(target_w / min(self.aspect_ratio_range)))
        elif in_ratio > max(self.aspect_ratio_range):
            target_h = h
            target_w = int(round(target_h * max(self.aspect_ratio_range)))
        else:  # whole image
            target_w = w
            target_h = h
        offset_h = (h - target_h) // 2
        offset_w = (w - target_w) // 2
        return offset_h, offset_w, target_h, target_w

    def _bbox_clip(self, bboxes: np.ndarray, img_shape: tuple[int, int]) -> np.ndarray:
        """Clip bboxes to fit the image shape.

        Copy from mmcv.image.geometric.bbox_clip

        Args:
            bboxes (ndarray): Shape (..., 4*k)
            img_shape (tuple[int]): (height, width) of the image.

        Returns:
            ndarray: Clipped bboxes.
        """
        cmin = np.empty(bboxes.shape[-1], dtype=bboxes.dtype)
        cmin[0::2] = img_shape[1] - 1
        cmin[1::2] = img_shape[0] - 1
        return np.maximum(np.minimum(bboxes, cmin), 0)

    def _bbox_scaling(self, bboxes: np.ndarray, scale: float, clip_shape: tuple[int, int] | None = None) -> np.ndarray:
        """Scaling bboxes w.r.t the box center.

        Copy from mmcv.image.geometric.bbox_scaling

        Args:
            bboxes (ndarray): Shape(..., 4).
            scale (float): Scaling factor.
            clip_shape (tuple[int], optional): If specified, bboxes that exceed the
                boundary will be clipped according to the given shape (h, w).

        Returns:
            ndarray: Scaled bboxes.
        """
        if float(scale) == 1.0:
            scaled_bboxes = bboxes.copy()
        else:
            w = bboxes[..., 2] - bboxes[..., 0] + 1
            h = bboxes[..., 3] - bboxes[..., 1] + 1
            dw = (w * (scale - 1)) * 0.5
            dh = (h * (scale - 1)) * 0.5
            scaled_bboxes = bboxes + np.stack((-dw, -dh, dw, dh), axis=-1)
        if clip_shape is not None:
            return self._bbox_clip(scaled_bboxes, clip_shape)
        return scaled_bboxes

    def _crop_img(
        self,
        img: np.ndarray,
        bboxes: np.ndarray,
        scale: float = 1.0,
        pad_fill: float | list | None = None,
    ) -> np.ndarray | list[np.ndarray]:
        """Crop image patches.

        Copy from mmcv.image.geometric.imcrop
        3 steps: scale the bboxes -> clip bboxes -> crop and pad.

        Args:
            img (ndarray): Image to be cropped.
            bboxes (ndarray): Shape (k, 4) or (4, ), location of cropped bboxes.
            scale (float, optional): Scale ratio of bboxes, the default value
                1.0 means no scaling.
            pad_fill (Number | list[Number]): Value to be filled for padding.
                Default: None, which means no padding.

        Returns:
            list[ndarray] | ndarray: The cropped image patches.
        """
        chn = 1 if img.ndim == 2 else img.shape[2]
        if pad_fill is not None and isinstance(pad_fill, (int, float)):
            pad_fill = [pad_fill for _ in range(chn)]

        _bboxes = bboxes[None, ...] if bboxes.ndim == 1 else bboxes
        scaled_bboxes = self._bbox_scaling(_bboxes, scale).astype(np.int32)
        clipped_bbox = self._bbox_clip(scaled_bboxes, img.shape)

        patches = []
        for i in range(clipped_bbox.shape[0]):
            x1, y1, x2, y2 = tuple(clipped_bbox[i, :])
            if pad_fill is None:
                patch = img[y1 : y2 + 1, x1 : x2 + 1, ...]
            else:
                _x1, _y1, _x2, _y2 = tuple(scaled_bboxes[i, :])
                patch_h = _y2 - _y1 + 1
                patch_w = _x2 - _x1 + 1
                patch_shape = (patch_h, patch_w) if chn == 1 else (patch_h, patch_w, chn)
                patch = np.array(pad_fill, dtype=img.dtype) * np.ones(patch_shape, dtype=img.dtype)
                x_start = 0 if _x1 >= 0 else -_x1
                y_start = 0 if _y1 >= 0 else -_y1
                w = x2 - x1 + 1
                h = y2 - y1 + 1
                patch[y_start : y_start + h, x_start : x_start + w, ...] = img[y1 : y1 + h, x1 : x1 + w, ...]
            patches.append(patch)

        if bboxes.ndim == 1:
            return patches[0]
        return patches

    def forward(self, *_inputs: T_OTXDataEntity) -> T_OTXDataEntity:
        """Transform function to randomly resized crop images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly resized cropped results, 'img_shape'
                key in result dict is updated according to crop size.
        """
        inputs = _inputs[0]
        if (img := getattr(inputs, "image", None)) is not None:
            img = to_np_image(img)
            offset_h, offset_w, target_h, target_w = self.rand_crop_params(img)
            bboxes = np.array(
                [
                    offset_w,
                    offset_h,
                    offset_w + target_w - 1,
                    offset_h + target_h - 1,
                ],
            )
            img = self._crop_img(img, bboxes=bboxes)
            inputs.img_info = _crop_image_info(inputs.img_info, *img.shape[:2])

            img = cv2.resize(
                img,
                tuple(self.scale[::-1]),
                dst=None,
                interpolation=self.cv2_interp_codes[self.interpolation],
            )
            if (masks := getattr(inputs, "gt_seg_map", None)) is not None:
                masks = masks.numpy()
                masks = self._crop_img(masks, bboxes=bboxes)
                masks = cv2.resize(
                    masks,
                    tuple(self.scale[::-1]),
                    dst=None,
                    interpolation=self.cv2_interp_codes["nearest"],
                )
                inputs.gt_seg_map = torch.from_numpy(masks)  # type: ignore[attr-defined]

            inputs.image = img
            inputs.img_info = _resize_image_info(inputs.img_info, img.shape[:2])
        return inputs

    def __repr__(self):
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__ + f"(scale={self.scale}"
        repr_str += ", crop_ratio_range="
        repr_str += f"{tuple(round(s, 4) for s in self.crop_ratio_range)}"
        repr_str += ", aspect_ratio_range="
        repr_str += f"{tuple(round(r, 4) for r in self.aspect_ratio_range)}"
        repr_str += f", max_attempts={self.max_attempts}"
        repr_str += f", interpolation={self.interpolation}"
        repr_str += f", backend={self.backend})"
        return repr_str


class RandomFlip(tvt_v2.Transform):
    """Implementation of mmdet.datasets.transforms.RandomFlip with torchvision format.

    Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L496-L596

    TODO : update masks for instance segmentation
    TODO : optimize logic to torcivision pipeline

     - ``prob`` is float, ``direction`` is string: the image will be
         ``direction``ly flipped with probability of ``prob`` .
         E.g., ``prob=0.5``, ``direction='horizontal'``,
         then image will be horizontally flipped with probability of 0.5.
     - ``prob`` is float, ``direction`` is list of string: the image will
         be ``direction[i]``ly flipped with probability of
         ``prob/len(direction)``.
         E.g., ``prob=0.5``, ``direction=['horizontal', 'vertical']``,
         then image will be horizontally flipped with probability of 0.25,
         vertically with probability of 0.25.
     - ``prob`` is list of float, ``direction`` is list of string:
         given ``len(prob) == len(direction)``, the image will
         be ``direction[i]``ly flipped with probability of ``prob[i]``.
         E.g., ``prob=[0.3, 0.5]``, ``direction=['horizontal',
         'vertical']``, then image will be horizontally flipped with
         probability of 0.3, vertically with probability of 0.5.

    Args:
         prob (float | list[float], optional): The flipping probability.
             Defaults to None.
         direction(str | list[str]): The flipping direction. Options
             If input is a list, the length must equal ``prob``. Each
             element in ``prob`` indicates the flip probability of
             corresponding direction. Defaults to 'horizontal'.
    """

    def __init__(
        self,
        prob: float | Iterable[float] | None = None,
        direction: str | Sequence[str | None] = "horizontal",
    ) -> None:
        super().__init__()

        if isinstance(prob, list):
            assert all(isinstance(p, float) for p in prob)  # noqa: S101
            assert 0 <= sum(prob) <= 1  # noqa: S101
        elif isinstance(prob, float):
            assert 0 <= prob <= 1  # noqa: S101
        else:
            msg = f"probs must be float or list of float, but got `{type(prob)}`."
            raise TypeError(msg)
        self.prob = prob

        valid_directions = ["horizontal", "vertical", "diagonal"]
        if isinstance(direction, str):
            assert direction in valid_directions  # noqa: S101
        elif isinstance(direction, list):
            assert all(isinstance(d, str) for d in direction)  # noqa: S101
            assert set(direction).issubset(set(valid_directions))  # noqa: S101
        else:
            msg = f"direction must be either str or list of str, but got `{type(direction)}`."
            raise TypeError(msg)
        self.direction = direction

        if isinstance(prob, list):
            assert len(prob) == len(self.direction)  # noqa: S101

    @cache_randomness
    def _choose_direction(self) -> str:
        """Choose the flip direction according to `prob` and `direction`."""
        if isinstance(self.direction, Sequence) and not isinstance(self.direction, str):
            # None means non-flip
            direction_list: list = [*list(self.direction), None]
        elif isinstance(self.direction, str):
            # None means non-flip
            direction_list = [self.direction, None]

        if isinstance(self.prob, list):
            non_prob: float = 1 - sum(self.prob)
            prob_list = [*self.prob, non_prob]
        elif isinstance(self.prob, float):
            non_prob = 1.0 - self.prob
            # exclude non-flip
            single_ratio = self.prob / (len(direction_list) - 1)
            prob_list = [single_ratio] * (len(direction_list) - 1) + [non_prob]

        return np.random.choice(direction_list, p=prob_list)

    def forward(self, *_inputs: T_OTXDataEntity) -> T_OTXDataEntity:
        """Flip images, bounding boxes, and semantic segmentation map."""
        assert len(_inputs) == 1, "[tmp] Multiple entity is not supported yet."  # noqa: S101
        inputs = _inputs[0]

        if (cur_dir := self._choose_direction()) is not None:
            # flip image
            img = to_np_image(inputs.image)
            img = np.ascontiguousarray(flip_image(img, direction=cur_dir))

            inputs.image = img

            # flip bboxes
            if hasattr(inputs, "bboxes") and (bboxes := getattr(inputs, "bboxes", None)) is not None:
                bboxes = flip_bboxes(bboxes, inputs.img_info.img_shape, direction=cur_dir)
                inputs.bboxes = tv_tensors.BoundingBoxes(bboxes, format="XYXY", canvas_size=img.shape[:2])

        return inputs

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(prob={self.prob}, "
        repr_str += f"direction={self.direction})"
        return repr_str


class PhotoMetricDistortion(tvt_v2.Transform):
    """Implementation of mmdet.datasets.transforms.PhotoMetricDistortion with torchvision format.

    Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L1084-L1210

    TODO : update masks for instance segmentation
    TODO : optimize logic to torcivision pipeline

    Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (sequence): range of contrast.
        saturation_range (sequence): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        brightness_delta: int = 32,
        contrast_range: Sequence[int | float] = (0.5, 1.5),
        saturation_range: Sequence[int | float] = (0.5, 1.5),
        hue_delta: int = 18,
    ) -> None:
        super().__init__()

        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    @cache_randomness
    def _random_flags(self) -> Sequence[int | float]:
        mode = random.randint(2)
        brightness_flag = random.randint(2)
        contrast_flag = random.randint(2)
        saturation_flag = random.randint(2)
        hue_flag = random.randint(2)
        swap_flag = random.randint(2)
        delta_value = random.uniform(-self.brightness_delta, self.brightness_delta)
        alpha_value = random.uniform(self.contrast_lower, self.contrast_upper)
        saturation_value = random.uniform(self.saturation_lower, self.saturation_upper)
        hue_value = random.uniform(-self.hue_delta, self.hue_delta)
        swap_value = random.permutation(3)

        return (
            mode,
            brightness_flag,
            contrast_flag,
            saturation_flag,
            hue_flag,
            swap_flag,
            delta_value,
            alpha_value,
            saturation_value,
            hue_value,
            swap_value,
        )

    def forward(self, *_inputs: T_OTXDataEntity) -> T_OTXDataEntity:
        """Transform function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """
        assert len(_inputs) == 1, "[tmp] Multiple entity is not supported yet."  # noqa: S101
        inputs = _inputs[0]

        if (img := getattr(inputs, "image", None)) is not None:
            img = to_np_image(img)
            img = img.astype(np.float32)

            (
                mode,
                brightness_flag,
                contrast_flag,
                saturation_flag,
                hue_flag,
                swap_flag,
                delta_value,
                alpha_value,
                saturation_value,
                hue_value,
                swap_value,
            ) = self._random_flags()

            # random brightness
            if brightness_flag:
                img += delta_value

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            if mode == 1 and contrast_flag:
                img *= alpha_value

            # TODO (sungchul): OTX consumes RGB images but mmx assumes they are BGR.
            # convert color from BGR to HSV
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # f32 -> f32

            # random saturation
            if saturation_flag:
                img[..., 1] *= saturation_value
                # For image(type=float32), after convert bgr to hsv by opencv,
                # valid saturation value range is [0, 1]
                if saturation_value > 1:
                    img[..., 1] = img[..., 1].clip(0, 1)

            # random hue
            if hue_flag:
                img[..., 0] += hue_value
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)  # f32 -> f32

            # random contrast
            if mode == 0 and contrast_flag:
                img *= alpha_value

            # randomly swap channels
            if swap_flag:
                img = img[..., swap_value]

            inputs.image = img  # f32
        return inputs

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(brightness_delta={self.brightness_delta}, "
        repr_str += "contrast_range="
        repr_str += f"{(self.contrast_lower, self.contrast_upper)}, "
        repr_str += "saturation_range="
        repr_str += f"{(self.saturation_lower, self.saturation_upper)}, "
        repr_str += f"hue_delta={self.hue_delta})"
        return repr_str


class RandomAffine(tvt_v2.Transform):
    """Implementation of mmdet.datasets.transforms.RandomAffine with torchvision format.

    Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L2736-L2901

    TODO : update masks for instance segmentation
    TODO : optimize logic to torcivision pipeline

    Args:
        max_rotate_degree (float): Maximum degrees of rotation transform.
            Defaults to 10.
        max_translate_ratio (float): Maximum ratio of translation.
            Defaults to 0.1.
        scaling_ratio_range (tuple[float]): Min and max ratio of
            scaling transform. Defaults to (0.5, 1.5).
        max_shear_degree (float): Maximum degrees of shear
            transform. Defaults to 2.
        border (tuple[int]): Distance from width and height sides of input
            image to adjust output shape. Only used in mosaic dataset.
            Defaults to (0, 0).
        border_val (tuple[int]): Border padding values of 3 channels.
            Defaults to (114, 114, 114).
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
    """

    def __init__(
        self,
        max_rotate_degree: float = 10.0,
        max_translate_ratio: float = 0.1,
        scaling_ratio_range: tuple[float, float] = (0.5, 1.5),
        max_shear_degree: float = 2.0,
        border: tuple[int, int] = (0, 0),
        border_val: tuple[int, int, int] = (114, 114, 114),
        bbox_clip_border: bool = True,
    ) -> None:
        super().__init__()

        assert 0 <= max_translate_ratio <= 1  # noqa: S101
        assert scaling_ratio_range[0] <= scaling_ratio_range[1]  # noqa: S101
        assert scaling_ratio_range[0] > 0  # noqa: S101
        self.max_rotate_degree = max_rotate_degree
        self.max_translate_ratio = max_translate_ratio
        self.scaling_ratio_range = scaling_ratio_range
        self.max_shear_degree = max_shear_degree
        self.border = border
        self.border_val = border_val
        self.bbox_clip_border = bbox_clip_border

    @cache_randomness
    def _get_random_homography_matrix(self, height: int, width: int) -> np.ndarray:
        # Rotation
        rotation_degree = random.uniform(-self.max_rotate_degree, self.max_rotate_degree)
        rotation_matrix = self._get_rotation_matrix(rotation_degree)

        # Scaling
        scaling_ratio = random.uniform(self.scaling_ratio_range[0], self.scaling_ratio_range[1])
        scaling_matrix = self._get_scaling_matrix(scaling_ratio)

        # Shear
        x_degree = random.uniform(-self.max_shear_degree, self.max_shear_degree)
        y_degree = random.uniform(-self.max_shear_degree, self.max_shear_degree)
        shear_matrix = self._get_shear_matrix(x_degree, y_degree)

        # Translation
        trans_x = random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * width
        trans_y = random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * height
        translate_matrix = self._get_translation_matrix(trans_x, trans_y)

        return translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix

    def forward(self, *_inputs: DetDataEntity) -> DetDataEntity:
        """Forward for RandomAffine."""
        assert len(_inputs) == 1, "[tmp] Multiple entity is not supported yet."  # noqa: S101
        inputs = _inputs[0]

        img = to_np_image(inputs.image)
        height = img.shape[0] + self.border[1] * 2
        width = img.shape[1] + self.border[0] * 2

        warp_matrix = self._get_random_homography_matrix(height, width)

        img = cv2.warpPerspective(img, warp_matrix, dsize=(width, height), borderValue=self.border_val)
        inputs.image = img
        inputs.img_info = _resize_image_info(inputs.img_info, img.shape[:2])

        bboxes = inputs.bboxes
        num_bboxes = len(bboxes)
        if num_bboxes:
            bboxes = project_bboxes(bboxes, warp_matrix)
            if self.bbox_clip_border:
                bboxes = clip_bboxes(bboxes, (height, width))
            # remove outside bbox
            valid_index = is_inside_bboxes(bboxes, (height, width))
            inputs.bboxes = tv_tensors.BoundingBoxes(bboxes[valid_index], format="XYXY", canvas_size=(height, width))
            inputs.labels = inputs.labels[valid_index]

        return inputs

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(max_rotate_degree={self.max_rotate_degree}, "
        repr_str += f"max_translate_ratio={self.max_translate_ratio}, "
        repr_str += f"scaling_ratio_range={self.scaling_ratio_range}, "
        repr_str += f"max_shear_degree={self.max_shear_degree}, "
        repr_str += f"border={self.border}, "
        repr_str += f"border_val={self.border_val}, "
        repr_str += f"bbox_clip_border={self.bbox_clip_border})"
        return repr_str

    @staticmethod
    def _get_rotation_matrix(rotate_degrees: float) -> np.ndarray:
        radian = math.radians(rotate_degrees)
        return np.array(
            [[np.cos(radian), -np.sin(radian), 0.0], [np.sin(radian), np.cos(radian), 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    @staticmethod
    def _get_scaling_matrix(scale_ratio: float) -> np.ndarray:
        return np.array([[scale_ratio, 0.0, 0.0], [0.0, scale_ratio, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    @staticmethod
    def _get_shear_matrix(x_shear_degrees: float, y_shear_degrees: float) -> np.ndarray:
        x_radian = math.radians(x_shear_degrees)
        y_radian = math.radians(y_shear_degrees)
        return np.array(
            [[1, np.tan(x_radian), 0.0], [np.tan(y_radian), 1, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    @staticmethod
    def _get_translation_matrix(x: float, y: float) -> np.ndarray:
        return np.array([[1, 0.0, x], [0.0, 1, y], [0.0, 0.0, 1.0]], dtype=np.float32)


class CachedMosaic(tvt_v2.Transform):
    """Implementation of mmdet.datasets.transforms.CachedMosaic with torchvision format.

    Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L3342-L3573

    TODO : update masks for instance segmentation
    TODO : optimize logic to torcivision pipeline

    Args:
        img_scale (Sequence[int]): Image size before mosaic pipeline of single
            image. The shape order should be (height, width).
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
        img_scale: tuple[int, int] | list[int] = (640, 640),  # (H, W)
        center_ratio_range: tuple[float, float] = (0.5, 1.5),
        bbox_clip_border: bool = True,
        pad_val: float = 114.0,
        prob: float = 1.0,
        max_cached_images: int = 40,
        random_pop: bool = True,
    ) -> None:
        super().__init__()

        assert isinstance(img_scale, (tuple, list))  # noqa: S101
        assert 0 <= prob <= 1.0, f"The probability should be in range [0,1]. got {prob}."  # noqa: S101

        self.img_scale = img_scale  # (H, W)
        self.center_ratio_range = center_ratio_range
        self.bbox_clip_border = bbox_clip_border
        self.pad_val = pad_val
        self.prob = prob

        self.results_cache: list[DetDataEntity] = []
        self.random_pop = random_pop
        assert max_cached_images >= 4, f"The length of cache must >= 4, but got {max_cached_images}."  # noqa: S101
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
        return [random.randint(0, len(cache) - 1) for _ in range(3)]

    def forward(self, *_inputs: DetDataEntity) -> DetDataEntity:
        """Forward for CachedMosaic."""
        assert len(_inputs) == 1, "[tmp] Multiple entity is not supported yet."  # noqa: S101
        inputs = _inputs[0]

        self.results_cache.append(copy.deepcopy(inputs))
        if len(self.results_cache) > self.max_cached_images:
            index = random.randint(0, len(self.results_cache) - 1) if self.random_pop else 0
            self.results_cache.pop(index)

        if len(self.results_cache) <= 4:
            return inputs

        if random.uniform(0, 1) > self.prob:
            return inputs

        indices = self.get_indexes(self.results_cache)
        mix_results = [copy.deepcopy(self.results_cache[i]) for i in indices]

        # TODO (mmdetection): refactor mosaic to reuse these code.
        # https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L3465
        mosaic_bboxes = []
        mosaic_bboxes_labels = []

        if len((inp_img := to_np_image(inputs.image)).shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), 3),
                self.pad_val,
                dtype=inp_img.dtype,
            )
        else:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                self.pad_val,
                dtype=inp_img.dtype,
            )

        # mosaic center x, y
        center_x = int(random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_y = int(random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_position = (center_x, center_y)

        loc_strs = ("top_left", "top_right", "bottom_left", "bottom_right")
        for i, loc in enumerate(loc_strs):
            results_patch = copy.deepcopy(inputs) if loc == "top_left" else copy.deepcopy(mix_results[i - 1])

            img_i = to_np_image(results_patch.image)
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[0] / h_i, self.img_scale[1] / w_i)
            img_i = cv2.resize(
                img_i,
                (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)),
                interpolation=cv2.INTER_LINEAR,
            )

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_bboxes_i = results_patch.bboxes
            gt_bboxes_labels_i = results_patch.labels

            padw = x1_p - x1_c
            padh = y1_p - y1_c
            gt_bboxes_i = rescale_bboxes(gt_bboxes_i, (scale_ratio_i, scale_ratio_i))
            gt_bboxes_i = translate_bboxes(gt_bboxes_i, (padw, padh))
            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_bboxes_labels.append(gt_bboxes_labels_i)

        mosaic_bboxes = torch.cat(mosaic_bboxes, dim=0)
        mosaic_bboxes_labels = torch.cat(mosaic_bboxes_labels, dim=0)

        if self.bbox_clip_border:
            mosaic_bboxes = clip_bboxes(mosaic_bboxes, (2 * self.img_scale[0], 2 * self.img_scale[1]))

        # remove outside bboxes
        inside_inds = is_inside_bboxes(mosaic_bboxes, (2 * self.img_scale[0], 2 * self.img_scale[1])).numpy()
        mosaic_bboxes = mosaic_bboxes[inside_inds]
        mosaic_bboxes_labels = mosaic_bboxes_labels[inside_inds]

        inputs.image = mosaic_img
        inputs.img_info = _resized_crop_image_info(
            inputs.img_info,
            mosaic_img.shape[:2],
        )  # TODO (sungchul): need to add proper function
        inputs.bboxes = tv_tensors.BoundingBoxes(mosaic_bboxes, format="XYXY", canvas_size=mosaic_img.shape[:2])
        inputs.labels = mosaic_bboxes_labels
        return inputs

    def _mosaic_combine(
        self,
        loc: str,
        center_position_xy: Sequence[float],
        img_shape_wh: Sequence[int],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Calculate global coordinate of mosaic image and local coordinate of cropped sub-image.

        Args:
            loc (str): Index for the sub-image, loc in ("top_left",
              "top_right", "bottom_left", "bottom_right").
            center_position_xy (Sequence[float]): Mixing center for 4 images,
                (x, y).
            img_shape_wh (Sequence[int]): Width and height of sub-image

        Returns:
            tuple[tuple[int]]: Corresponding coordinate of pasting and
                cropping
                - paste_coord (tuple): paste corner coordinate in mosaic image.
                - crop_coord (tuple): crop corner coordinate in mosaic image.
        """
        assert loc in ("top_left", "top_right", "bottom_left", "bottom_right")  # noqa: S101
        if loc == "top_left":
            # index0 to top left part of image
            x1, y1, x2, y2 = map(
                int,
                (
                    max(center_position_xy[0] - img_shape_wh[0], 0),
                    max(center_position_xy[1] - img_shape_wh[1], 0),
                    center_position_xy[0],
                    center_position_xy[1],
                ),
            )
            crop_coord = img_shape_wh[0] - (x2 - x1), img_shape_wh[1] - (y2 - y1), img_shape_wh[0], img_shape_wh[1]

        elif loc == "top_right":
            # index1 to top right part of image
            x1, y1, x2, y2 = map(
                int,
                (
                    center_position_xy[0],
                    max(center_position_xy[1] - img_shape_wh[1], 0),
                    min(center_position_xy[0] + img_shape_wh[0], self.img_scale[1] * 2),
                    center_position_xy[1],
                ),
            )
            crop_coord = 0, img_shape_wh[1] - (y2 - y1), min(img_shape_wh[0], x2 - x1), img_shape_wh[1]

        elif loc == "bottom_left":
            # index2 to bottom left part of image
            x1, y1, x2, y2 = map(
                int,
                (
                    max(center_position_xy[0] - img_shape_wh[0], 0),
                    center_position_xy[1],
                    center_position_xy[0],
                    min(self.img_scale[0] * 2, center_position_xy[1] + img_shape_wh[1]),
                ),
            )
            crop_coord = img_shape_wh[0] - (x2 - x1), 0, img_shape_wh[0], min(y2 - y1, img_shape_wh[1])

        else:
            # index3 to bottom right part of image
            x1, y1, x2, y2 = map(
                int,
                (
                    center_position_xy[0],
                    center_position_xy[1],
                    min(center_position_xy[0] + img_shape_wh[0], self.img_scale[1] * 2),
                    min(self.img_scale[0] * 2, center_position_xy[1] + img_shape_wh[1]),
                ),
            )
            crop_coord = 0, 0, min(img_shape_wh[0], x2 - x1), min(y2 - y1, img_shape_wh[1])

        paste_coord = x1, y1, x2, y2
        return paste_coord, crop_coord

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(img_scale={self.img_scale}, "
        repr_str += f"center_ratio_range={self.center_ratio_range}, "
        repr_str += f"pad_val={self.pad_val}, "
        repr_str += f"prob={self.prob}, "
        repr_str += f"max_cached_images={self.max_cached_images}, "
        repr_str += f"random_pop={self.random_pop})"
        return repr_str


class CachedMixUp(tvt_v2.Transform):
    """Implementation of mmdet.datasets.transforms.CachedMixup with torchvision format.

    Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L3577-L3854

    TODO : update masks for instance segmentation
    TODO : optimize logic to torcivision pipeline

    Args:
        img_scale (Sequence[int]): Image output size after mixup pipeline.
            The shape order should be (height, width). Defaults to (640, 640).
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

    def __init__(
        self,
        img_scale: tuple[int, int] | list[int] = (640, 640),  # (H, W)
        ratio_range: tuple[float, float] = (0.5, 1.5),
        flip_ratio: float = 0.5,
        pad_val: float = 114.0,
        max_iters: int = 15,
        bbox_clip_border: bool = True,
        max_cached_images: int = 20,
        random_pop: bool = True,
        prob: float = 1.0,
    ) -> None:
        super().__init__()

        assert isinstance(img_scale, (tuple, list))  # noqa: S101
        assert max_cached_images >= 2, f"The length of cache must >= 2, but got {max_cached_images}."  # noqa: S101
        assert 0 <= prob <= 1.0, f"The probability should be in range [0,1]. got {prob}."  # noqa: S101
        self.dynamic_scale = img_scale  # (H, W)
        self.ratio_range = ratio_range
        self.flip_ratio = flip_ratio
        self.pad_val = pad_val
        self.max_iters = max_iters
        self.bbox_clip_border = bbox_clip_border
        self.results_cache: list[DetDataEntity] = []

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
        for _ in range(self.max_iters):
            index = random.randint(0, len(cache) - 1)
            gt_bboxes_i = cache[index].bboxes
            if len(gt_bboxes_i) != 0:
                break
        return index

    def forward(self, *_inputs: DetDataEntity) -> DetDataEntity:
        """MixUp transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        # cache and pop images
        assert len(_inputs) == 1, "[tmp] Multiple entity is not supported yet."  # noqa: S101
        inputs = _inputs[0]

        self.results_cache.append(copy.deepcopy(inputs))
        if len(self.results_cache) > self.max_cached_images:
            index = random.randint(0, len(self.results_cache) - 1) if self.random_pop else 0
            self.results_cache.pop(index)

        if len(self.results_cache) <= 1:
            return inputs

        if random.uniform(0, 1) > self.prob:
            return inputs

        index = self.get_indexes(self.results_cache)
        retrieve_results = copy.deepcopy(self.results_cache[index])

        # TODO (mmdetection): refactor mixup to reuse these code.
        # https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L3721
        if retrieve_results.bboxes.shape[0] == 0:
            # empty bbox
            return inputs

        retrieve_img = to_np_image(retrieve_results.image)

        jit_factor = random.uniform(*self.ratio_range)
        is_flip = random.uniform(0, 1) > self.flip_ratio

        if len(retrieve_img.shape) == 3:
            out_img = (
                np.ones((self.dynamic_scale[0], self.dynamic_scale[1], 3), dtype=retrieve_img.dtype) * self.pad_val
            )
        else:
            out_img = np.ones(self.dynamic_scale, dtype=retrieve_img.dtype) * self.pad_val

        # 1. keep_ratio resize
        scale_ratio = min(self.dynamic_scale[0] / retrieve_img.shape[0], self.dynamic_scale[1] / retrieve_img.shape[1])
        retrieve_img = cv2.resize(
            retrieve_img,
            (int(retrieve_img.shape[1] * scale_ratio), int(retrieve_img.shape[0] * scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )

        # 2. paste
        out_img[: retrieve_img.shape[0], : retrieve_img.shape[1]] = retrieve_img

        # 3. scale jit
        scale_ratio *= jit_factor
        out_img = cv2.resize(
            out_img,
            (int(out_img.shape[1] * jit_factor), int(out_img.shape[0] * jit_factor)),
            interpolation=cv2.INTER_LINEAR,
        )

        # 4. flip
        if is_flip:
            out_img = out_img[:, ::-1, :]

        # 5. random crop
        ori_img = to_np_image(inputs.image)
        origin_h, origin_w = out_img.shape[:2]
        target_h, target_w = ori_img.shape[:2]
        padded_img = np.ones((max(origin_h, target_h), max(origin_w, target_w), 3)) * self.pad_val
        padded_img[:origin_h, :origin_w] = out_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w)
        padded_cropped_img = padded_img[y_offset : y_offset + target_h, x_offset : x_offset + target_w]

        # 6. adjust bbox
        retrieve_gt_bboxes = retrieve_results.bboxes
        retrieve_gt_bboxes = rescale_bboxes(retrieve_gt_bboxes, (scale_ratio, scale_ratio))

        if self.bbox_clip_border:
            retrieve_gt_bboxes = clip_bboxes(retrieve_gt_bboxes, (origin_h, origin_w))

        if is_flip:
            retrieve_gt_bboxes = flip_bboxes(retrieve_gt_bboxes, (origin_h, origin_w), direction="horizontal")

        # 7. filter
        cp_retrieve_gt_bboxes = retrieve_gt_bboxes.clone()
        cp_retrieve_gt_bboxes = translate_bboxes(cp_retrieve_gt_bboxes, (-x_offset, -y_offset))

        if self.bbox_clip_border:
            cp_retrieve_gt_bboxes = clip_bboxes(cp_retrieve_gt_bboxes, (target_h, target_w))

        # 8. mix up
        ori_img = ori_img.astype(np.float32)
        mixup_img = 0.5 * ori_img + 0.5 * padded_cropped_img.astype(np.float32)

        retrieve_gt_bboxes_labels = retrieve_results.labels

        mixup_gt_bboxes = torch.cat((inputs.bboxes, cp_retrieve_gt_bboxes), dim=0)
        mixup_gt_bboxes_labels = torch.cat((inputs.labels, retrieve_gt_bboxes_labels), dim=0)

        # remove outside bbox
        inside_inds = is_inside_bboxes(mixup_gt_bboxes, (target_h, target_w))
        mixup_gt_bboxes = mixup_gt_bboxes[inside_inds]
        mixup_gt_bboxes_labels = mixup_gt_bboxes_labels[inside_inds]

        inputs.image = mixup_img.astype(np.uint8)
        inputs.img_info = _resized_crop_image_info(
            inputs.img_info,
            mixup_img.shape[:2],
        )  # TODO (sungchul): need to add proper function
        inputs.bboxes = tv_tensors.BoundingBoxes(mixup_gt_bboxes, format="XYXY", canvas_size=mixup_img.shape[:2])
        inputs.labels = mixup_gt_bboxes_labels
        return inputs

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(dynamic_scale={self.dynamic_scale}, "
        repr_str += f"ratio_range={self.ratio_range}, "
        repr_str += f"flip_ratio={self.flip_ratio}, "
        repr_str += f"pad_val={self.pad_val}, "
        repr_str += f"max_iters={self.max_iters}, "
        repr_str += f"bbox_clip_border={self.bbox_clip_border}, "
        repr_str += f"max_cached_images={self.max_cached_images}, "
        repr_str += f"random_pop={self.random_pop}, "
        repr_str += f"prob={self.prob})"
        return repr_str


class YOLOXHSVRandomAug(tvt_v2.Transform):
    """Implementation of mmdet.datasets.transforms.YOLOXHSVRandomAug with torchvision format.

    Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L2905-L2961

    TODO : update masks for instance segmentation
    TODO : optimize logic to torcivision pipeline

    Args:
        hue_delta (int): delta of hue. Defaults to 5.
        saturation_delta (int): delta of saturation. Defaults to 30.
        value_delta (int): delat of value. Defaults to 30.
    """

    def __init__(self, hue_delta: int = 5, saturation_delta: int = 30, value_delta: int = 30) -> None:
        super().__init__()

        self.hue_delta = hue_delta
        self.saturation_delta = saturation_delta
        self.value_delta = value_delta

    @cache_randomness
    def _get_hsv_gains(self) -> np.ndarray:
        hsv_gains = np.random.uniform(-1, 1, 3) * [
            self.hue_delta,
            self.saturation_delta,
            self.value_delta,
        ]
        # random selection of h, s, v
        hsv_gains *= random.randint(0, 2, 3)
        # prevent overflow
        return hsv_gains.astype(np.int16)

    def forward(self, *_inputs: T_OTXDataEntity) -> T_OTXDataEntity:
        """Forward for random hsv transform."""
        assert len(_inputs) == 1, "[tmp] Multiple entity is not supported yet."  # noqa: S101
        inputs = _inputs[0]

        img = to_np_image(inputs.image)
        hsv_gains = self._get_hsv_gains()
        # TODO (sungchul): OTX det models except for YOLOX-S, L, X consume RGB images but mmdet assumes they are BGR.
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

        img_hsv[..., 0] = (img_hsv[..., 0] + hsv_gains[0]) % 180
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_gains[1], 0, 255)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_gains[2], 0, 255)
        cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)

        inputs.image = img
        return inputs

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(hue_delta={self.hue_delta}, "
        repr_str += f"saturation_delta={self.saturation_delta}, "
        repr_str += f"value_delta={self.value_delta})"
        return repr_str


class Pad(tvt_v2.Transform):
    """Implementation of mmdet.datasets.transforms.Pad with torchvision format.

    Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/datasets/transforms/transforms.py#L705-L784

    TODO : update masks for instance segmentation
    TODO : optimize logic to torcivision pipeline

    Args:
        size (tuple, optional): Fixed padding size.
            Expected padding shape (width, height). Defaults to None.
        size_divisor (int, optional): The divisor of padded size. Defaults to
            None.
        pad_to_square (bool): Whether to pad the image into a square.
            Currently only used for YOLOX. Defaults to False.
        pad_val (Number | dict[str, Number], optional) - Padding value for if
            the pad_mode is "constant".  If it is a single number, the value
            to pad the image is the number and to pad the semantic
            segmentation map is 255. If it is a dict, it should have the
            following keys:

            - img: The value to pad the image.
            - seg: The value to pad the semantic segmentation map.
            Defaults to dict(img=0, seg=255).
        padding_mode (str): Type of padding. Should be: constant, edge,
            reflect or symmetric. Defaults to 'constant'.

            - constant: pads with a constant value, this value is specified
              with pad_val.
            - edge: pads with the last value at the edge of the image.
            - reflect: pads with reflection of image without repeating the last
              value on the edge. For example, padding [1, 2, 3, 4] with 2
              elements on both sides in reflect mode will result in
              [3, 2, 1, 2, 3, 4, 3, 2].
            - symmetric: pads with reflection of image repeating the last value
              on the edge. For example, padding [1, 2, 3, 4] with 2 elements on
              both sides in symmetric mode will result in
              [2, 1, 1, 2, 3, 4, 4, 3]
    """

    border_type: ClassVar = {
        "constant": cv2.BORDER_CONSTANT,
        "edge": cv2.BORDER_REPLICATE,
        "reflect": cv2.BORDER_REFLECT_101,
        "symmetric": cv2.BORDER_REFLECT,
    }

    def __init__(
        self,
        size: tuple[int, int] | None = None,  # (H, W)
        size_divisor: int | None = None,
        pad_to_square: bool = False,
        pad_val: int | float | dict | None = None,
        padding_mode: str = "constant",
    ) -> None:
        super().__init__()

        self.size = size
        self.size_divisor = size_divisor
        pad_val = pad_val or {"img": 0, "seg": 255}
        if isinstance(pad_val, int):
            pad_val = {"img": pad_val, "seg": 255}
        assert isinstance(pad_val, dict), "pad_val "  # noqa: S101
        self.pad_val = pad_val
        self.pad_to_square = pad_to_square

        if pad_to_square:
            assert size is None, "The size and size_divisor must be None when pad2square is True"  # noqa: S101
        else:
            assert size is not None or size_divisor is not None, "only one of size and size_divisor should be valid"  # noqa: S101
            assert size is None or size_divisor is None  # noqa: S101
        assert padding_mode in ["constant", "edge", "reflect", "symmetric"]  # noqa: S101
        self.padding_mode = padding_mode

    def _pad_img(self, inputs: T_OTXDataEntity) -> T_OTXDataEntity:
        """Pad images according to ``self.size``."""
        img = to_np_image(inputs.image)
        pad_val = self.pad_val.get("img", 0)

        size: tuple[int, int]
        if self.pad_to_square:
            max_size = max(img.shape[:2])
            size = (max_size, max_size)

        if self.size_divisor is not None:
            if not self.pad_to_square:
                size = (img.shape[0], img.shape[1])
            pad_h = int(np.ceil(size[0] / self.size_divisor)) * self.size_divisor
            pad_w = int(np.ceil(size[1] / self.size_divisor)) * self.size_divisor
            size = (pad_h, pad_w)
        elif self.size is not None:
            size = self.size

        if isinstance(pad_val, int) and img.ndim == 3:
            pad_val = tuple(pad_val for _ in range(img.shape[2]))

        width = max(size[1] - img.shape[1], 0)
        height = max(size[0] - img.shape[0], 0)
        padding = [0, 0, width, height]

        padded_img = img = cv2.copyMakeBorder(
            img,
            padding[1],
            padding[3],
            padding[0],
            padding[2],
            self.border_type[self.padding_mode],
            value=pad_val,
        )

        inputs.image = padded_img
        inputs.img_info = _pad_image_info(inputs.img_info, padding)
        return inputs

    def forward(self, *_inputs: T_OTXDataEntity) -> T_OTXDataEntity:
        """Call function to pad images."""
        assert len(_inputs) == 1, "[tmp] Multiple entity is not supported yet."  # noqa: S101
        inputs = _inputs[0]

        return self._pad_img(inputs)


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
