# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX base data entities."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, Generic, Iterator, TypeVar

import torch
import torchvision.transforms.v2.functional as F  # noqa: N812
from torch import Tensor, stack
from torch.utils._pytree import tree_flatten
from torchvision import tv_tensors

from otx.core.data.entity.utils import clamp_points, register_pytree_node
from otx.core.types.image import ImageColorChannel, ImageType
from otx.core.types.task import OTXTaskType

if TYPE_CHECKING:
    import decord
    import numpy as np


def custom_wrap(wrappee: Tensor, *, like: tv_tensors.TVTensor, **kwargs) -> tv_tensors.TVTensor:
    """Add `Points` in tv_tensors.wrap.

    If `like` is
        - tv_tensors.BoundingBoxes : the `format` and `canvas_size` of `like` are assigned to `wrappee`
        - Points : the `canvas_size` of `like` is assigned to `wrappee`
    Unless, they are passed as `kwargs`.

    Args:
        wrappee (Tensor): The tensor to convert.
        like (tv_tensors.TVTensor): The reference. `wrappee` will be converted into the same subclass as `like`.
        kwargs: Can contain "format" and "canvas_size" if `like` is a tv_tensor.BoundingBoxes,
            or "canvas_size" if `like` is a `Points`. Ignored otherwise.
    """
    if isinstance(like, tv_tensors.BoundingBoxes):
        return tv_tensors.BoundingBoxes._wrap(  # noqa: SLF001
            wrappee,
            format=kwargs.get("format", like.format),
            canvas_size=kwargs.get("canvas_size", like.canvas_size),
        )
    elif isinstance(like, Points):  # noqa: RET505
        return Points._wrap(wrappee, canvas_size=kwargs.get("canvas_size", like.canvas_size))  # noqa: SLF001

    # TODO(Vlad): remove this after torch upgrade. This workaround prevents a failure when like is also a Tensor
    if type(like) == type(wrappee):
        return wrappee

    return wrappee.as_subclass(type(like))


tv_tensors.wrap = custom_wrap


class ImageInfo(tv_tensors.TVTensor):
    """Meta info for image.

    Attributes:
        img_id: Image id
        img_shape: Image shape (heigth, width) after preprocessing
        ori_shape: Image shape (heigth, width) right after loading it
        padding: Number of pixels to pad all borders (left, top, right, bottom)
        scale_factor: Scale factor (height, width) if the image is resized during preprocessing.
            Default value is `(1.0, 1.0)` when there is no resizing. However, if the image is cropped,
            it will lose the scaling information and be `None`.
        normalized: If true, this image is normalized with `norm_mean` and `norm_std`
        norm_mean: Mean vector used to normalize this image
        norm_std: Standard deviation vector used to normalize this image
        image_color_channel: Color channel type of this image, RGB or BGR.
        ignored_labels: Label that should be ignored in this image. Default to None.
    """

    img_idx: int
    img_shape: tuple[int, int]
    ori_shape: tuple[int, int]
    padding: tuple[int, int, int, int] = (0, 0, 0, 0)
    scale_factor: tuple[float, float] | None = (1.0, 1.0)
    normalized: bool = False
    norm_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    norm_std: tuple[float, float, float] = (1.0, 1.0, 1.0)
    image_color_channel: ImageColorChannel = ImageColorChannel.RGB
    ignored_labels: list[int]

    @classmethod
    def _wrap(
        cls,
        dummy_tensor: Tensor,
        *,
        img_idx: int,
        img_shape: tuple[int, int],
        ori_shape: tuple[int, int],
        padding: tuple[int, int, int, int] = (0, 0, 0, 0),
        scale_factor: tuple[float, float] | None = (1.0, 1.0),
        normalized: bool = False,
        norm_mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
        norm_std: tuple[float, float, float] = (1.0, 1.0, 1.0),
        image_color_channel: ImageColorChannel = ImageColorChannel.RGB,
        ignored_labels: list[int] | None = None,
    ) -> ImageInfo:
        image_info = dummy_tensor.as_subclass(cls)
        image_info.img_idx = img_idx
        image_info.img_shape = img_shape
        image_info.ori_shape = ori_shape
        image_info.padding = padding
        image_info.scale_factor = scale_factor
        image_info.normalized = normalized
        image_info.norm_mean = norm_mean
        image_info.norm_std = norm_std
        image_info.image_color_channel = image_color_channel
        image_info.ignored_labels = ignored_labels if ignored_labels else []
        return image_info

    def __new__(  # noqa: D102
        cls,
        img_idx: int,
        img_shape: tuple[int, int],
        ori_shape: tuple[int, int],
        padding: tuple[int, int, int, int] = (0, 0, 0, 0),
        scale_factor: tuple[float, float] | None = (1.0, 1.0),
        normalized: bool = False,
        norm_mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
        norm_std: tuple[float, float, float] = (1.0, 1.0, 1.0),
        image_color_channel: ImageColorChannel = ImageColorChannel.RGB,
        ignored_labels: list[int] | None = None,
    ) -> ImageInfo:
        return cls._wrap(
            dummy_tensor=Tensor(),
            img_idx=img_idx,
            img_shape=img_shape,
            ori_shape=ori_shape,
            padding=padding,
            scale_factor=scale_factor,
            normalized=normalized,
            norm_mean=norm_mean,
            norm_std=norm_std,
            image_color_channel=image_color_channel,
            ignored_labels=ignored_labels,
        )

    @classmethod
    def _wrap_output(
        cls,
        output: Tensor,
        args: tuple[()] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> ImageType:
        """Wrap an output (`torch.Tensor`) obtained from PyTorch function.

        For example, this function will be called when

        >>> img_info = ImageInfo(img_idx=0, img_shape=(10, 10), ori_shape=(10, 10))
        >>> `_wrap_output()` will be called after the PyTorch function `to()` is called
        >>> img_info = img_info.to(device=torch.cuda)
        """
        flat_params, _ = tree_flatten(args + (tuple(kwargs.values()) if kwargs else ()))

        if isinstance(output, Tensor) and not isinstance(output, ImageInfo):
            image_info = next(x for x in flat_params if isinstance(x, ImageInfo))
            output = ImageInfo._wrap(
                dummy_tensor=output,
                img_idx=image_info.img_idx,
                img_shape=image_info.img_shape,
                ori_shape=image_info.ori_shape,
                padding=image_info.padding,
                scale_factor=image_info.scale_factor,
                normalized=image_info.normalized,
                norm_mean=image_info.norm_mean,
                norm_std=image_info.norm_std,
                image_color_channel=image_info.image_color_channel,
                ignored_labels=image_info.ignored_labels,
            )
        elif isinstance(output, (tuple, list)):
            image_infos = [x for x in flat_params if isinstance(x, ImageInfo)]
            output = type(output)(
                ImageInfo._wrap(
                    dummy_tensor=dummy_tensor,
                    img_idx=image_info.img_idx,
                    img_shape=image_info.img_shape,
                    ori_shape=image_info.ori_shape,
                    padding=image_info.padding,
                    scale_factor=image_info.scale_factor,
                    normalized=image_info.normalized,
                    norm_mean=image_info.norm_mean,
                    norm_std=image_info.norm_std,
                    image_color_channel=image_info.image_color_channel,
                    ignored_labels=image_info.ignored_labels,
                )
                for dummy_tensor, image_info in zip(output, image_infos)
            )
        return output

    def __repr__(self) -> str:
        return (
            "ImageInfo("
            f"img_idx={self.img_idx}, "
            f"img_shape={self.img_shape}, "
            f"ori_shape={self.ori_shape}, "
            f"padding={self.padding}, "
            f"scale_factor={self.scale_factor}, "
            f"normalized={self.normalized}, "
            f"norm_mean={self.norm_mean}, "
            f"norm_std={self.norm_std}, "
            f"image_color_channel={self.image_color_channel}, "
            f"ignored_labels={self.ignored_labels})"
        )


class VideoInfo(tv_tensors.TVTensor):
    """Meta info for video.

    Attributes:
        clip_len: Length of a video clip.
        num_clips: Number of clips for training.
        frame_interval: Interval between sampled frames in a video clip.
        video_reader: Decord video reader.
        avg_fps: Average number of frames per seconds in a clip.
        num_frames: Number of total frames in a video clip.
        start_index: Start frame index.
        frame_inds: Numpy array of chosen frame indices.
    """

    clip_len: int
    num_clips: int
    frame_interval: int
    video_reader: decord.VideoReader
    avg_fps: float
    num_frames: int
    start_index: int
    frame_inds: np.ndarray

    @classmethod
    def _wrap(
        cls,
        dummy_tensor: Tensor,
        *,
        clip_len: int = 8,
        num_clips: int = 1,
        frame_interval: int = 4,
        video_reader: decord.VideoReader | None = None,
        avg_fps: float = 30.0,
        num_frames: int | None = None,
        start_index: int = 0,
        frame_inds: np.ndarray | None = None,
    ) -> ImageInfo:
        video_info = dummy_tensor.as_subclass(cls)
        video_info.video_reader = video_reader
        video_info.avg_fps = avg_fps
        video_info.num_frames = num_frames
        video_info.clip_len = clip_len
        video_info.num_clips = num_clips
        video_info.frame_interval = frame_interval
        video_info.start_index = start_index
        video_info.frame_inds = frame_inds
        return video_info

    def __new__(  # noqa: D102
        cls,
        clip_len: int = 8,
        num_clips: int = 1,
        frame_interval: int = 4,
        video_reader: decord.VideoReader | None = None,
        avg_fps: float = 30.0,
        num_frames: int | None = None,
        start_index: int = 0,
        frame_inds: np.ndarray | None = None,
    ) -> VideoInfo:
        return cls._wrap(
            dummy_tensor=Tensor(),
            clip_len=clip_len,
            num_clips=num_clips,
            frame_interval=frame_interval,
            video_reader=video_reader,
            avg_fps=avg_fps,
            num_frames=num_frames,
            start_index=start_index,
            frame_inds=frame_inds,
        )

    @classmethod
    def _wrap_output(
        cls,
        output: Tensor,
        args: tuple[()] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> VideoInfo | list[VideoInfo] | tuple[VideoInfo]:
        """Wrap an output (`torch.Tensor`) obtained from PyTorch function.

        For example, this function will be called when

        >>> img_info = VideoInfo(img_idx=0, img_shape=(10, 10), ori_shape=(10, 10))
        >>> `_wrap_output()` will be called after the PyTorch function `to()` is called
        >>> img_info = img_info.to(device=torch.cuda)
        """
        flat_params, _ = tree_flatten(args + (tuple(kwargs.values()) if kwargs else ()))

        if isinstance(output, Tensor) and not isinstance(output, VideoInfo):
            video_info = next(x for x in flat_params if isinstance(x, VideoInfo))
            output = VideoInfo._wrap(
                dummy_tensor=output,
                clip_len=video_info.clip_len,
                num_clips=video_info.num_clips,
                frame_interval=video_info.frame_interval,
                video_reader=video_info.video_reader,
                avg_fps=video_info.avg_fps,
                num_frames=video_info.num_frames,
                start_index=video_info.start_index,
                frame_inds=video_info.frame_inds,
            )
        elif isinstance(output, (tuple, list)):
            video_infos = [x for x in flat_params if isinstance(x, VideoInfo)]
            output = type(output)(
                VideoInfo._wrap(
                    dummy_tensor=dummy_tensor,
                    clip_len=video_info.clip_len,
                    num_clips=video_info.num_clips,
                    frame_interval=video_info.frame_interval,
                    video_reader=video_info.video_reader,
                    avg_fps=video_info.avg_fps,
                    num_frames=video_info.num_frames,
                    start_index=video_info.start_index,
                    frame_inds=video_info.frame_inds,
                )
                for dummy_tensor, video_info in zip(output, video_infos)
            )
        return output

    def __repr__(self) -> str:
        return (
            "VideoInfo("
            f"clip_len={self.clip_len}, "
            f"num_clips={self.num_clips}, "
            f"frame_interval={self.frame_interval}, "
            f"video_reader={self.video_reader}, "
            f"avg_fps={self.avg_fps}, "
            f"num_frames={self.num_frames}, "
            f"start_index={self.start_index}, "
            f"frame_inds={self.frame_inds})"
        )


class BboxInfo(tv_tensors.TVTensor):
    """`torch.Tensor` subclass for bbox info, e.g., center, scale, and rotation.

    Attributes:
        data: Any data that can be turned into a tensor with `torch.as_tensor`.
        center (np.ndarray): Bbox center coordinates.
        scale (np.ndarray): Bbox scales for width and height.
        rotation (float): Bbox rotation for bbox augmentations.
    """

    center: np.ndarray
    scale: np.ndarray
    rotation: float

    @classmethod
    def _wrap(
        cls,
        dummy_tensor: Tensor,
        *,
        center: np.ndarray | None = None,
        scale: np.ndarray | None = None,
        rotation: float | None = None,
    ) -> BboxInfo:
        bbox_info = dummy_tensor.as_subclass(cls)
        bbox_info.center = center
        bbox_info.scale = scale
        bbox_info.rotation = rotation
        return bbox_info

    def __new__(  # noqa: D102
        cls,
        center: np.ndarray | None = None,
        scale: np.ndarray | None = None,
        rotation: float | None = None,
    ) -> BboxInfo:
        return cls._wrap(
            dummy_tensor=Tensor(),
            center=center,
            scale=scale,
            rotation=rotation,
        )

    @classmethod
    def _wrap_output(
        cls,
        output: Tensor,
        args: tuple[()] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> BboxInfo:
        flat_params, _ = tree_flatten(args + (tuple(kwargs.values()) if kwargs else ()))

        if isinstance(output, Tensor) and not isinstance(output, BboxInfo):
            bbox_info = next(x for x in flat_params if isinstance(x, BboxInfo))
            output = BboxInfo._wrap(
                dummy_tensor=output,
                center=bbox_info.center,
                scale=bbox_info.scale,
                rotation=bbox_info.rotation,
            )
        elif isinstance(output, (tuple, list)):
            bbox_infos = [x for x in flat_params if isinstance(x, BboxInfo)]
            output = type(output)(
                BboxInfo._wrap(
                    dummy_tensor=dummy_tensor,
                    center=bbox_info.center,
                    scale=bbox_info.scale,
                    rotation=bbox_info.rotation,
                )
                for dummy_tensor, bbox_info in zip(output, bbox_infos)
            )
        return output

    def __repr__(self, *, tensor_contents: Any = None) -> str:  # noqa: ANN401
        return f"BboxInfo(center={self.center}, scale={self.scale}, rotation={self.rotation})"


@F.register_kernel(functional=F.resize, tv_tensor_cls=ImageInfo)
def _resize_image_info(image_info: ImageInfo, size: list[int], **kwargs) -> ImageInfo:  # noqa: ARG001
    """Register ImageInfo to TorchVision v2 resize kernel."""
    if len(size) == 2:
        image_info.img_shape = (size[0], size[1])
    elif len(size) == 1:
        image_info.img_shape = (size[0], size[0])
    else:
        raise ValueError(size)

    ori_h, ori_w = image_info.ori_shape
    new_h, new_w = image_info.img_shape
    image_info.scale_factor = (new_h / ori_h, new_w / ori_w)
    return image_info


@F.register_kernel(functional=F.crop, tv_tensor_cls=ImageInfo)
def _crop_image_info(
    image_info: ImageInfo,
    height: int,
    width: int,
    **kwargs,  # noqa: ARG001
) -> ImageInfo:
    """Register ImageInfo to TorchVision v2 resize kernel."""
    image_info.img_shape = (height, width)
    image_info.scale_factor = None
    return image_info


@F.register_kernel(functional=F.resized_crop, tv_tensor_cls=ImageInfo)
def _resized_crop_image_info(
    image_info: ImageInfo,
    size: list[int],
    **kwargs,  # noqa: ARG001
) -> ImageInfo:
    """Register ImageInfo to TorchVision v2 resize kernel."""
    if len(size) == 2:
        image_info.img_shape = (size[0], size[1])
    elif len(size) == 1:
        image_info.img_shape = (size[0], size[0])
    else:
        raise ValueError(size)

    image_info.scale_factor = None
    return image_info


@F.register_kernel(functional=F.center_crop, tv_tensor_cls=ImageInfo)
def _center_crop_image_info(
    image_info: ImageInfo,
    output_size: list[int],
    **kwargs,  # noqa: ARG001
) -> ImageInfo:
    """Register ImageInfo to TorchVision v2 resize kernel."""
    img_shape = F._geometry._center_crop_parse_output_size(output_size)  # noqa: SLF001
    image_info.img_shape = (img_shape[0], img_shape[1])

    image_info.scale_factor = None
    return image_info


@F.register_kernel(functional=F.pad, tv_tensor_cls=ImageInfo)
def _pad_image_info(
    image_info: ImageInfo,
    padding: int | list[int],
    **kwargs,  # noqa: ARG001
) -> ImageInfo:
    """Register ImageInfo to TorchVision v2 resize kernel."""
    left, right, top, bottom = F._geometry._parse_pad_padding(padding)  # noqa: SLF001
    height, width = image_info.img_shape
    image_info.padding = (left, top, right, bottom)
    image_info.img_shape = (height + top + bottom, width + left + right)
    return image_info


@F.register_kernel(functional=F.normalize, tv_tensor_cls=ImageInfo)
def _normalize_image_info(
    image_info: ImageInfo,
    mean: list[float],
    std: list[float],
    **kwargs,  # noqa: ARG001
) -> ImageInfo:
    image_info.normalized = True
    image_info.norm_mean = (mean[0], mean[1], mean[2])
    image_info.norm_std = (std[0], std[1], std[2])
    return image_info


class Points(tv_tensors.TVTensor):
    """`torch.Tensor` subclass for points.

    Attributes:
        data: Any data that can be turned into a tensor with `torch.as_tensor`.
        canvas_size (two-tuple of ints): Height and width of the corresponding image or video.
        dtype (torch.dtype, optional): Desired data type of the point. If omitted, will be inferred from `data`.
        device (torch.device, optional): Desired device of the point. If omitted and `data` is a
            `torch.Tensor`, the device is taken from it. Otherwise, the point is constructed on the CPU.
        requires_grad (bool, optional): Whether autograd should record operations on the point. If omitted and
            `data` is a `torch.Tensor`, the value is taken from it. Otherwise, defaults to `False`.
    """

    canvas_size: tuple[int, int]

    @classmethod
    def _wrap(cls, tensor: Tensor, *, canvas_size: tuple[int, int]) -> Points:
        points = tensor.as_subclass(cls)
        points.canvas_size = canvas_size
        return points

    def __new__(  # noqa: D102
        cls,
        data: Any,  # noqa: ANN401
        *,
        canvas_size: tuple[int, int],
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool | None = None,
    ) -> Points:
        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        return cls._wrap(tensor, canvas_size=canvas_size)

    @classmethod
    def _wrap_output(
        cls,
        output: Tensor,
        args: tuple[()] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> Points:
        flat_params, _ = tree_flatten(args + (tuple(kwargs.values()) if kwargs else ()))
        first_point_from_args = next(x for x in flat_params if isinstance(x, Points))
        canvas_size = first_point_from_args.canvas_size

        if isinstance(output, Tensor) and not isinstance(output, Points):
            output = Points._wrap(output, canvas_size=canvas_size)
        elif isinstance(output, (tuple, list)):
            output = type(output)(Points._wrap(part, canvas_size=canvas_size) for part in output)
        return output

    def __repr__(self, *, tensor_contents: Any = None) -> str:  # noqa: ANN401
        return self._make_repr(canvas_size=self.canvas_size)


def resize_points(
    points: torch.Tensor,
    canvas_size: tuple[int, int],
    size: tuple[int, int] | list[int],
    max_size: int | None = None,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Resize points."""
    old_height, old_width = canvas_size
    new_height, new_width = F._geometry._compute_resized_output_size(  # noqa: SLF001
        canvas_size,
        size=size,
        max_size=max_size,
    )

    if (new_height, new_width) == (old_height, old_width):
        return points, canvas_size

    w_ratio = new_width / old_width
    h_ratio = new_height / old_height
    ratios = torch.tensor([w_ratio, h_ratio], device=points.device)
    return (
        points.mul(ratios).to(points.dtype),
        (new_height, new_width),
    )


@F.register_kernel(functional=F.resize, tv_tensor_cls=Points)
def _resize_points_dispatch(
    inpt: Points,
    size: tuple[int, int] | list[int],
    max_size: int | None = None,
    **kwargs,  # noqa: ARG001
) -> Points:
    output, canvas_size = resize_points(
        inpt.as_subclass(torch.Tensor),
        inpt.canvas_size,
        size,
        max_size=max_size,
    )
    return tv_tensors.wrap(output, like=inpt, canvas_size=canvas_size)


def pad_points(
    points: torch.Tensor,
    canvas_size: tuple[int, int],
    padding: tuple[int, ...] | list[int],
    padding_mode: str = "constant",
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad points."""
    if padding_mode not in ["constant"]:
        # TODO(sungchul): add support of other padding modes
        raise ValueError(f"Padding mode '{padding_mode}' is not supported with bounding boxes")  # noqa: EM102, TRY003

    left, right, top, bottom = F._geometry._parse_pad_padding(padding)  # noqa: SLF001

    pad = [left, top]
    points = points + torch.tensor(pad, dtype=points.dtype, device=points.device)

    height, width = canvas_size
    height += top + bottom
    width += left + right
    canvas_size = (height, width)

    return clamp_points(points, canvas_size=canvas_size), canvas_size


@F.register_kernel(functional=F.pad, tv_tensor_cls=Points)
def _pad_points_dispatch(
    inpt: Points,
    padding: tuple[int, ...] | list[int],
    padding_mode: str = "constant",
    **kwargs,  # noqa: ARG001
) -> Points:
    output, canvas_size = pad_points(
        inpt.as_subclass(torch.Tensor),
        canvas_size=inpt.canvas_size,
        padding=padding,
        padding_mode=padding_mode,
    )
    return tv_tensors.wrap(output, like=inpt, canvas_size=canvas_size)


@F.register_kernel(functional=F.get_size, tv_tensor_cls=Points)
def get_size_points(point: Points) -> list[int]:
    """Get size of points."""
    return list(point.canvas_size)


T_OTXDataEntity = TypeVar(
    "T_OTXDataEntity",
    bound="OTXDataEntity",
)


@register_pytree_node
@dataclass
class OTXDataEntity(Mapping):
    """Base data entity for OTX.

    This entity is the output of each OTXDataset,
    which can be go through the input preprocessing tranforms.

    :param task: OTX task definition
    :param image: Image tensor or list of Image tensor which can have different type according to `image_type`
        1) `image_type=ImageType.NUMPY`: H x W x C numpy image tensor
        2) `image_type=ImageType.TV_IMAGE`: C x H x W torchvision image tensor
        3) `image_type=ImageType.NUMPY_LIST`: List of H x W x C numpy image tensors
        3) `image_type=ImageType.TV_IMAGE_LIST`: List of C x H x W torchvision image tensors
    :param imgs_info: Meta information for images
    """

    image: np.ndarray | tv_tensors.Image | list[np.ndarray] | list[tv_tensors.Image]
    img_info: ImageInfo

    @property
    def task(self) -> OTXTaskType:
        """OTX task type definition."""
        msg = "OTXTaskType is not defined."
        raise RuntimeError(msg)

    @property
    def image_type(self) -> ImageType:
        """Image type definition."""
        return ImageType.get_image_type(self.image)

    def to_tv_image(self: T_OTXDataEntity) -> T_OTXDataEntity:
        """Return a new instance with the `image` attribute converted to a TorchVision Image if it is a NumPy array.

        Returns:
            A new instance with the `image` attribute converted to a TorchVision Image, if applicable.
            Otherwise, return this instance as is.
        """
        if isinstance(self.image, tv_tensors.Image):
            return self

        return self.wrap(image=F.to_image(self.image))

    def __iter__(self) -> Iterator[str]:
        for field_ in fields(self):
            yield field_.name

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        return getattr(self, key)

    def __len__(self) -> int:
        """Get the number of fields in this data entity."""
        return len(fields(self))

    def wrap(self: T_OTXDataEntity, **kwargs) -> T_OTXDataEntity:
        """Wrap this dataclass with the given keyword arguments.

        Args:
            **kwargs: Keyword arguments to be overwritten on top of this dataclass
        Returns:
            Updated dataclass
        """
        updated_kwargs = asdict(self)
        updated_kwargs.update(**kwargs)
        return self.__class__(**updated_kwargs)


@dataclass
class OTXPredEntity(OTXDataEntity):
    """Data entity to represent the model output prediction."""

    score: np.ndarray | Tensor

    saliency_map: np.ndarray | Tensor | None = None
    feature_vector: np.ndarray | list | None = None


T_OTXBatchDataEntity = TypeVar(
    "T_OTXBatchDataEntity",
    bound="OTXBatchDataEntity",
)


@dataclass
class OTXBatchDataEntity(Generic[T_OTXDataEntity]):
    """Base Batch data entity for OTX.

    This entity is the output of PyTorch DataLoader,
    which is the direct input of OTXModel.

    :param images: List of B numpy RGB image tensors (C x H x W) or
        An image tensor stacked with B RGB image tensors (B x C x H x W)
    :param imgs_info: Meta information for images
    """

    batch_size: int
    images: list[tv_tensors.Image] | tv_tensors.Image
    imgs_info: list[ImageInfo]

    @property
    def task(self) -> OTXTaskType:
        """OTX task type definition."""
        msg = "OTXTaskType is not defined."
        raise RuntimeError(msg)

    @property
    def images_type(self) -> ImageType:
        """Images type definition."""
        return ImageType.get_image_type(self.images)

    @classmethod
    def collate_fn(
        cls,
        entities: list[T_OTXDataEntity],
        stack_images: bool = True,
    ) -> OTXBatchDataEntity:
        """Collection function to collect `OTXDataEntity` into `OTXBatchDataEntity` in data loader.

        Args:
            entities: List of OTX data entities.
            stack_images: If True, return 4D B x C x H x W image tensor.
                Otherwise return a list of 3D C x H x W image tensor.

        Returns:
            Collated OTX batch data entity
        """
        if (batch_size := len(entities)) == 0:
            msg = "collate_fn() input should have > 0 entities"
            raise RuntimeError(msg)

        task = entities[0].task

        if not all(task == entity.task for entity in entities):
            msg = "collate_fn() input should include a single OTX task"
            raise RuntimeError(msg)

        images = [entity.image for entity in entities]
        like = next(iter(images))

        if stack_images and not all(like.shape == entity.image.shape for entity in entities):  # type: ignore[union-attr]
            msg = (
                "You set stack_images as True, but not all images in the batch has same shape. "
                "In this case, we cannot stack images. Some tasks, e.g., detection, "
                "can have different image shapes among samples in the batch. However, if it is not your intention, "
                "consider setting stack_images as False in the config."
            )
            warnings.warn(msg, stacklevel=1)
            stack_images = False

        return OTXBatchDataEntity(
            batch_size=batch_size,
            images=tv_tensors.wrap(torch.stack(images), like=like) if stack_images else images,
            imgs_info=[entity.img_info for entity in entities],
        )

    @property
    def stacked_images(self) -> tv_tensors.Image:
        """A stacked image tensor (B x C x H x W).

        if its `images` field is a list of image tensors,
        convert it to a 4D image tensor and return it.
        Otherwise, return it as is.
        """
        if isinstance(self.images, tv_tensors.Image):
            return self.images

        like = next(iter(self.images))
        return tv_tensors.wrap(stack(self.images), like=like)

    def pin_memory(self: T_OTXBatchDataEntity) -> T_OTXBatchDataEntity:
        """Pin memory for member tensor variables."""
        # TODO(vinnamki): Keep track this issue
        # https://github.com/pytorch/pytorch/issues/116403
        return self.wrap(
            images=(
                [tv_tensors.wrap(image.pin_memory(), like=image) for image in self.images]
                if isinstance(self.images, list)
                else tv_tensors.wrap(self.images.pin_memory(), like=self.images)
            ),
        )

    def wrap(self: T_OTXBatchDataEntity, **kwargs) -> T_OTXBatchDataEntity:
        """Wrap this dataclass with the given keyword arguments.

        Args:
            **kwargs: Keyword arguments to be overwritten on top of this dataclass
        Returns:
            Updated dataclass
        """
        updated_kwargs = asdict(self)
        updated_kwargs.update(**kwargs)
        return self.__class__(**updated_kwargs)


@dataclass
class OTXBatchPredEntity(OTXBatchDataEntity):
    """Data entity to represent model output predictions.

    Attributes:
        scores: List of probability scores representing model predictions.
        saliency_map: List of saliency maps used to explain model predictions.
            This field is optional and will be an empty list for non-XAI pipelines.
        feature_vector: List of intermediate feature vectors used for model predictions.
            This field is optional and will be an empty list for non-XAI pipelines.
    """

    scores: list[np.ndarray] | list[Tensor]

    # (Optional) XAI-related outputs
    # TODO(vinnamkim): These are actually plural, but their namings are not
    # This is because ModelAPI requires the OV IR to produce `saliency_map`
    # and `feature_vector` (singular) named outputs.
    # It should be fixed later.
    saliency_map: list[np.ndarray] | list[Tensor] = field(default_factory=list)
    feature_vector: list[np.ndarray] | list[Tensor] = field(default_factory=list)

    @property
    def has_xai_outputs(self) -> bool:
        """If the XAI related fields are fulfilled, return True."""
        # NOTE: Don't know why but some of test cases in tests/integration/api/test_xai.py
        # produce `len(self.saliency_map) > 0` and `len(self.feature_vector) == 0`
        # return len(self.saliency_map) > 0 and len(self.feature_vector) > 0
        return len(self.saliency_map) > 0


class OTXBatchLossEntity(Dict[str, Tensor]):
    """Data entity to represent model output losses."""


T_OTXBatchPredEntity = TypeVar(
    "T_OTXBatchPredEntity",
    bound=OTXBatchPredEntity,
)

T_OTXBatchLossEntity = TypeVar(
    "T_OTXBatchLossEntity",
    bound=OTXBatchLossEntity,
)
