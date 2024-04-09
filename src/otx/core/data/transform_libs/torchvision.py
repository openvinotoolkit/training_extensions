# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Helper to support TorchVision data transform functions."""

from __future__ import annotations

import copy
from inspect import isclass
from typing import TYPE_CHECKING, Any, Sequence

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
from torchvision.transforms.v2.functional._color import (_hsv_to_rgb,
                                                         _rgb_to_hsv)

from otx.core.data.entity.action_classification import ActionClsDataEntity
from otx.core.data.entity.base import (Points, _crop_image_info,
                                       _resize_image_info)
from otx.core.data.transform_libs.utils import (_scale_size, cache_randomness,
                                                centers_bboxes, clip_bboxes,
                                                flip_bboxes, is_inside_bboxes,
                                                overlap_bboxes, rescale_bboxes,
                                                rescale_size, to_np_image,
                                                translate_bboxes)

if TYPE_CHECKING:
    from torchvision.transforms.v2 import Compose

    from otx.core.config.data import SubsetConfig
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

    def __init__(self,
                 min_ious: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
                 min_crop_size: float = 0.3,
                 bbox_clip_border: bool = True) -> None:
        super().__init__()
        self.min_ious = min_ious
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size
        self.bbox_clip_border = bbox_clip_border

    @cache_randomness
    def _random_mode(self) -> int | float:
        return random.choice(self.sample_mode)

    def forward(self, *inputs: Any) -> Any:
        assert len(inputs) == 1, "[tmp] Multiple entity is not supported yet."
        inputs = inputs[0]

        img = to_np_image(inputs.image)
        boxes = inputs.bboxes
        h, w, c = img.shape
        while True:
            mode = self._random_mode()
            self.mode = mode
            if mode == 1:
                return inputs

            min_iou = self.mode
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array(
                    (int(left), int(top), int(left + new_w), int(top + new_h)))
                # Line or point crop is not allowed
                if patch[2] == patch[0] or patch[3] == patch[1]:
                    continue
                overlaps = overlap_bboxes(
                    torch.as_tensor(patch.reshape(-1, 4).astype(np.float32)),
                    boxes).numpy().reshape(-1)
                if len(overlaps) > 0 and overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                # only adjust boxes and instance masks when the gt is not empty
                if len(overlaps) > 0:
                    # adjust boxes
                    def is_center_of_bboxes_in_patch(boxes, patch):
                        centers = centers_bboxes(boxes).numpy()
                        mask = ((centers[:, 0] > patch[0]) *
                                (centers[:, 1] > patch[1]) *
                                (centers[:, 0] < patch[2]) *
                                (centers[:, 1] < patch[3]))
                        return mask

                    mask = is_center_of_bboxes_in_patch(boxes, patch)
                    if not mask.any():
                        continue
                    if hasattr(inputs, "bboxes"):
                        boxes = inputs.bboxes
                        mask = is_center_of_bboxes_in_patch(boxes, patch)
                        boxes = boxes[mask]
                        boxes = translate_bboxes(boxes, [-patch[0], -patch[1]])
                        if self.bbox_clip_border:
                            boxes = clip_bboxes(boxes, [patch[3] - patch[1], patch[2] - patch[0]])
                        inputs.bboxes = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(patch[3] - patch[1], patch[2] - patch[0]))

                        # labels
                        if hasattr(inputs, "labels"):
                            inputs.labels = inputs.labels[mask]

                # adjust the img no matter whether the gt is empty before crop
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                inputs.image = F.to_image(img)
                inputs.img_info = _crop_image_info(inputs.img_info, patch[3] - patch[1], patch[2] - patch[0])
                return inputs

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(min_ious={self.min_ious}, '
        repr_str += f'min_crop_size={self.min_crop_size}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
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
    cv2_interp_codes = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
        'area': cv2.INTER_AREA,
        'lanczos': cv2.INTER_LANCZOS4
    }

    def __init__(self,
                 scale: int | tuple[int, int] = None,
                 scale_factor: float | tuple[float, float] = None,
                 keep_ratio: bool = False,
                 clip_object_border: bool = True,
                 interpolation: str = "bilinear") -> None:
        super().__init__()

        assert scale is not None or scale_factor is not None, (
            '`scale` and'
            '`scale_factor` can not both be `None`')
        if scale is None:
            self.scale = None
        else:
            if isinstance(scale, int):
                self.scale = (scale, scale)
            else:
                self.scale = scale

        self.interpolation = interpolation
        self.keep_ratio = keep_ratio
        self.clip_object_border = clip_object_border
        if scale_factor is None:
            self.scale_factor = None
        elif isinstance(scale_factor, float):
            self.scale_factor = (scale_factor, scale_factor)
        elif isinstance(scale_factor, tuple):
            assert (len(scale_factor)) == 2
            self.scale_factor = scale_factor
        else:
            raise TypeError(
                f'expect scale_factor is float or Tuple(float), but'
                f'get {type(scale_factor)}')

    def _resize_img(self, inputs: DetDataEntity) -> DetDataEntity:
        """Resize images with inputs.img_info.img_shape."""
        if hasattr(inputs, "image"):
            img = to_np_image(inputs.image)
            if self.keep_ratio:
                h, w = img.shape[:2]
                new_size, scale_factor = rescale_size((w, h), inputs.img_info.img_shape[::-1], return_scale=True)
                img = cv2.resize(img, new_size, interpolation=self.cv2_interp_codes[self.interpolation])
            else:
                img = cv2.resize(img, inputs.img_info.img_shape[::-1], interpolation=self.cv2_interp_codes[self.interpolation])

            inputs.image = F.to_image(img)
            inputs.img_info = _resize_image_info(inputs.img_info, img.shape[:2])
        return inputs

    def _resize_bboxes(self, inputs: DetDataEntity) -> DetDataEntity:
        """Resize bounding boxes with inputs.img_info.scale_factor."""
        if hasattr(inputs, "bboxes"):
            bboxes = inputs.bboxes
            bboxes = rescale_bboxes(bboxes, inputs.img_info.scale_factor[::-1]) # HW -> WH
            if self.clip_object_border:
                bboxes = clip_bboxes(bboxes, inputs.img_info.img_shape)
            inputs.bboxes = tv_tensors.BoundingBoxes(bboxes, format="XYXY", canvas_size=inputs.img_info.img_shape)
        return inputs

    def forward(self, *inputs: Any) -> Any:
        """Transform function to resize images and bounding boxes.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'scale', 'scale_factor', 'height', 'width', and 'keep_ratio' keys
            are updated in result dict.
        """
        assert len(inputs) == 1, "[tmp] Multiple entity is not supported yet."
        inputs = inputs[0]

        if self.scale:
            inputs.img_info = _resize_image_info(inputs.img_info, self.scale)
        else:
            img_shape = to_np_image(inputs.image).shape[:2]
            inputs.img_info = _resize_image_info(inputs.img_info, _scale_size(img_shape[::-1], self.scale_factor))

        inputs = self._resize_img(inputs)
        inputs = self._resize_bboxes(inputs)

        return inputs

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.scale}, '
        repr_str += f'scale_factor={self.scale_factor}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'clip_object_border={self.clip_object_border}), '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


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

        self.img_scale = img_scale # (H, W)
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

        if len((inp_img := to_np_image(inputs.image)).shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), 3),
                self.pad_val,
                dtype=inp_img.dtype)
        else:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                self.pad_val,
                dtype=inp_img.dtype)

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

            img_i = to_np_image(results_patch.image)
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[0] / h_i,
                                self.img_scale[1] / w_i)
            img_i = cv2.resize(img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)), interpolation=cv2.INTER_LINEAR)

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

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
        inside_inds = is_inside_bboxes(mosaic_bboxes, [2 * self.img_scale[0], 2 * self.img_scale[1]]).numpy()
        mosaic_bboxes = mosaic_bboxes[inside_inds]
        mosaic_bboxes_labels = mosaic_bboxes_labels[inside_inds]

        inputs.image = F.to_image(mosaic_img)
        inputs.img_info = _resize_image_info(inputs.img_info, mosaic_img.shape[:2])
        inputs.bboxes = tv_tensors.BoundingBoxes(mosaic_bboxes, format="XYXY", canvas_size=mosaic_img.shape[:2])
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
                                 self.img_scale[1] * 2), \
                             center_position_xy[1]
            crop_coord = 0, img_shape_wh[1] - (y2 - y1), min(
                img_shape_wh[0], x2 - x1), img_shape_wh[1]

        elif loc == "bottom_left":
            # index2 to bottom left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             center_position_xy[1], \
                             center_position_xy[0], \
                             min(self.img_scale[0] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = img_shape_wh[0] - (x2 - x1), 0, img_shape_wh[0], min(
                y2 - y1, img_shape_wh[1])

        else:
            # index3 to bottom right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             center_position_xy[1], \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[1] * 2), \
                             min(self.img_scale[0] * 2, center_position_xy[1] +
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

    def __init__(self,
                 img_scale: tuple[int, int] | list[int] = (640, 640), # (H, W)
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
        self.dynamic_scale = img_scale # (H, W)
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

        retrieve_img = to_np_image(retrieve_results.image)

        jit_factor = random.uniform(*self.ratio_range)
        is_flip = random.uniform(0, 1) > self.flip_ratio

        if len(retrieve_img.shape) == 3:
            out_img = np.ones(
                (self.dynamic_scale[0], self.dynamic_scale[1], 3),
                dtype=retrieve_img.dtype) * self.pad_val
        else:
            out_img = np.ones(
                self.dynamic_scale,
                dtype=retrieve_img.dtype) * self.pad_val

        # 1. keep_ratio resize
        scale_ratio = min(self.dynamic_scale[0] / retrieve_img.shape[0],
                          self.dynamic_scale[1] / retrieve_img.shape[1])
        retrieve_img = cv2.resize(retrieve_img, (int(retrieve_img.shape[1] * scale_ratio), int(retrieve_img.shape[0] * scale_ratio)), interpolation=cv2.INTER_LINEAR)

        # 2. paste
        out_img[:retrieve_img.shape[0], :retrieve_img.shape[1]] = retrieve_img

        # 3. scale jit
        scale_ratio *= jit_factor
        out_img = cv2.resize(out_img, (int(out_img.shape[1] * jit_factor), int(out_img.shape[0] * jit_factor)), interpolation=cv2.INTER_LINEAR)

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
        padded_cropped_img = padded_img[y_offset:y_offset + target_h, x_offset:x_offset + target_w]

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
        ori_img = ori_img.astype(np.float32)
        mixup_img = 0.5 * ori_img + 0.5 * padded_cropped_img.astype(np.float32)

        retrieve_gt_bboxes_labels = retrieve_results.labels

        mixup_gt_bboxes = torch.cat((inputs.bboxes, cp_retrieve_gt_bboxes), dim=0)
        mixup_gt_bboxes_labels = torch.cat((inputs.labels, retrieve_gt_bboxes_labels), dim=0)

        # remove outside bbox
        inside_inds = is_inside_bboxes(mixup_gt_bboxes, [target_h, target_w])
        mixup_gt_bboxes = mixup_gt_bboxes[inside_inds]
        mixup_gt_bboxes_labels = mixup_gt_bboxes_labels[inside_inds]

        inputs.image = F.to_image(mixup_img.astype(np.uint8))
        inputs.img_info = _resize_image_info(inputs.img_info, mixup_img.shape[:2])
        inputs.bboxes = tv_tensors.BoundingBoxes(mixup_gt_bboxes, format="XYXY", canvas_size=mixup_img.shape[:2])
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
