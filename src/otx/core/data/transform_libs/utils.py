# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Utils for data transform functions."""

from __future__ import annotations

import copy
import functools
import inspect
import itertools
import weakref
from typing import TYPE_CHECKING, Sequence

import cv2
import numpy as np
import torch
from shapely import geometry
from torch import BoolTensor, Tensor

if TYPE_CHECKING:
    from datumaro import Polygon


CV2_INTERP_CODES = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


class cache_randomness:  # noqa: N801
    """Decorator that marks the method with random return value(s) in a transform class.

    Reference : https://github.com/open-mmlab/mmcv/blob/v2.1.0/mmcv/transforms/utils.py#L15-L87

    This decorator is usually used together with the context-manager
    :func`:cache_random_params`. In this context, a decorated method will
    cache its return value(s) at the first time of being invoked, and always
    return the cached values when being invoked again.

    .. note::
        Only an instance method can be decorated with ``cache_randomness``.
    """

    def __init__(self, func):  # noqa: ANN001
        # Check `func` is to be bound as an instance method
        if not inspect.isfunction(func):
            msg = "Unsupport callable to decorate with@cache_randomness."
            raise TypeError(msg)
        func_args = inspect.getfullargspec(func).args
        if len(func_args) == 0 or func_args[0] != "self":
            msg = (
                "@cache_randomness should only be used to decorate instance methods (the first argument is ``self``).",
            )
            raise TypeError(msg)

        functools.update_wrapper(self, func)
        self.func = func
        self.instance_ref = None

    def __set_name__(self, owner, name):  # noqa: ANN001
        # Maintain a record of decorated methods in the class
        if not hasattr(owner, "_methods_with_randomness"):
            owner._methods_with_randomness = []  # noqa: SLF001

        # Here `name` equals to `self.__name__`, i.e., the name of the
        # decorated function, due to the invocation of `update_wrapper` in
        # `self.__init__()`
        owner._methods_with_randomness.append(name)  # noqa: SLF001

    def __call__(self, *args, **kwargs):  # noqa: D102
        # Get the transform instance whose method is decorated
        # by cache_randomness
        instance = self.instance_ref()
        name = self.__name__

        # Check the flag ``self._cache_enabled``, which should be
        # set by the contextmanagers like ``cache_random_parameters```
        cache_enabled = getattr(instance, "_cache_enabled", False)

        if cache_enabled:
            # Initialize the cache of the transform instances. The flag
            # ``cache_enabled``` is set by contextmanagers like
            # ``cache_random_params```.
            if not hasattr(instance, "_cache"):
                instance._cache = {}  # noqa: SLF001

            if name not in instance._cache:  # noqa: SLF001
                instance._cache[name] = self.func(instance, *args, **kwargs)  # noqa: SLF001
            # Return the cached value
            return instance._cache[name]  # noqa: SLF001

        # Clear cache
        if hasattr(instance, "_cache"):
            del instance._cache  # noqa: SLF001
        # Return function output
        return self.func(instance, *args, **kwargs)

    def __get__(self, obj, cls):  # noqa: ANN001
        self.instance_ref = weakref.ref(obj)
        # Return a copy to avoid multiple transform instances sharing
        # one `cache_randomness` instance, which may cause data races
        # in multithreading cases.
        return copy.copy(self)


def get_image_shape(img: np.ndarray | Tensor | list) -> tuple[int, int]:
    """Get image(s) shape with (height, width)."""
    if not isinstance(img, (np.ndarray, Tensor, list)):
        msg = f"{type(img)} is not supported."
        raise TypeError(msg)

    if isinstance(img, np.ndarray):
        return img.shape[:2]
    if isinstance(img, Tensor):
        return img.shape[-2:]
    return get_image_shape(img[0])  # for list


def to_np_image(img: np.ndarray | Tensor | list) -> np.ndarray | list[np.ndarray]:
    """Convert torch.Tensor 3D image to numpy 3D image.

    TODO (sungchul): move it into base data entity?

    """
    if isinstance(img, np.ndarray):
        return img
    if isinstance(img, list):
        return [to_np_image(im) for im in img]
    return np.ascontiguousarray(img.numpy().transpose(1, 2, 0))


def rescale_bboxes(boxes: Tensor, scale_factor: tuple[float, float]) -> Tensor:
    """Rescale boxes w.r.t. rescale_factor in-place.

    Note:
        Both ``rescale_`` and ``resize_`` will enlarge or shrink boxes
        w.r.t ``scale_facotr``. The difference is that ``resize_`` only
        changes the width and the height of boxes, but ``rescale_`` also
        rescales the box centers simultaneously.

    Args:
        boxes (Tensor): bounding boxes to be rescaled.
        scale_factor (tuple[float, float]): factors for scaling boxes with (height, width).
            It will be used after flipped. The length should be 2.

    Returns:
        (Tensor): rescaled bounding boxes.
    """
    assert len(scale_factor) == 2  # noqa: S101
    scale_factor = boxes.new_tensor(scale_factor[::-1]).repeat(2)
    return boxes * scale_factor


def rescale_masks(
    masks: np.ndarray,
    scale_factor: float | tuple[float, float],  # (H, W)
    interpolation: str = "nearest",
) -> np.ndarray:
    """Rescale masks as large as possible while keeping the aspect ratio.

    Args:
        masks (np.ndarray): Masks to be rescaled.
        scale_factor (float | tuple[float, float]): Scale factor to be applied to masks with (height, width).
        interpolation (str): Interpolation mode. Defaults to `nearest`.

    Returns:
        (np.ndarray) : The rescaled masks.
    """
    h, w = masks.shape[1:]
    new_size = rescale_size((h, w), scale_factor)  # (H, W)

    # flipping `new_size` is required because cv2.resize uses (W, H)
    return np.stack(
        [cv2.resize(mask, new_size[::-1], interpolation=CV2_INTERP_CODES[interpolation]) for mask in masks],
    )


def rescale_polygons(polygons: list[Polygon], scale_factor: float | tuple[float, float]) -> list[Polygon]:
    """Rescale polygons as large as possible while keeping the aspect ratio.

    Args:
        polygons (np.ndarray): Polygons to be rescaled.
        scale_factor (float | tuple[float, float]): Scale factor to be applied to polygons with (height, width)
            or single float value.

    Returns:
        (np.ndarray) : The rescaled polygons.
    """
    if isinstance(scale_factor, float):
        w_scale = h_scale = scale_factor
    else:
        h_scale, w_scale = scale_factor

    for polygon in polygons:
        p = np.asarray(polygon.points, dtype=np.float32)
        p[0::2] *= w_scale
        p[1::2] *= h_scale
        polygon.points = p.tolist()
    return polygons


def translate_bboxes(boxes: Tensor, distances: Sequence[float]) -> Tensor:
    """Translate boxes in-place.

    Args:
        boxes (Tensor): Bounding boxes to be translated.
        distances (Sequence[float]): Translate distances. The first
            is horizontal distance and the second is vertical distance.

    Returns:
        (Tensor): Translated bounding boxes.
    """
    assert len(distances) == 2  # noqa: S101
    return boxes + boxes.new_tensor(distances).repeat(2)


def translate_masks(
    masks: np.ndarray,
    out_shape: tuple[int, int],
    offset: int | float,
    direction: str = "horizontal",
    border_value: int | tuple[int] = 0,
    interpolation: str = "bilinear",
) -> np.ndarray:
    """Translate the masks.

    Args:
        masks (np.ndarray): Masks to be translated.
        out_shape (tuple[int]): Shape for output mask, format (h, w).
        offset (int | float): The offset for translate.
        direction (str): The translate direction, either "horizontal" or "vertical".
        border_value (int | tuple[int]): Border value. Default 0 for masks.
        interpolation (str): Interpolation method, accepted values are
            'nearest', 'bilinear', 'bicubic', 'area', 'lanczos'. Defaults to
            'bilinear'.

    Returns:
        (np.ndarray): Translated BitmapMasks.
    """
    dtype = masks.dtype
    if masks.shape[-2:] != out_shape:
        empty_masks = np.zeros((masks.shape[0], *out_shape), dtype=dtype)
        min_h = min(out_shape[0], masks.shape[1])
        min_w = min(out_shape[1], masks.shape[2])
        empty_masks[:, :min_h, :min_w] = masks[:, :min_h, :min_w]
        masks = empty_masks

    # from https://github.com/open-mmlab/mmcv/blob/v2.1.0/mmcv/image/geometric.py#L740-L788
    height, width = masks.shape[1:]
    if masks.ndim == 2:
        channels = 1
    elif masks.ndim == 3:
        channels = masks.shape[0]

    if isinstance(border_value, int):
        border_value = tuple([border_value] * channels)  # type: ignore[assignment]
    elif isinstance(border_value, tuple):
        assert len(border_value) == channels, (  # noqa: S101
            "Expected the num of elements in tuple equals the channels"
            f"of input image. Found {len(border_value)} vs {channels}"
        )
    else:
        msg = f"Invalid type {type(border_value)} for `border_value`."
        raise ValueError(msg)  # noqa: TRY004

    translate_matrix = _get_translate_matrix(offset, direction)
    translated_masks = cv2.warpAffine(
        masks.transpose((1, 2, 0)),
        translate_matrix,
        (width, height),
        # Note case when the number elements in `border_value`
        # greater than 3 (e.g. translating masks whose channels
        # large than 3) will raise TypeError in `cv2.warpAffine`.
        # Here simply slice the first 3 values in `border_value`.
        borderValue=border_value[:3],  # type: ignore[index]
        flags=CV2_INTERP_CODES[interpolation],
    )

    if translated_masks.ndim == 2:
        translated_masks = translated_masks[:, :, None]
    return translated_masks.transpose((2, 0, 1)).astype(dtype)


def translate_polygons(
    polygons: list[Polygon],
    out_shape: tuple[int, int],
    offset: int | float,
    direction: str = "horizontal",
    border_value: int | float = 0,
) -> list[Polygon]:
    """Translate polygons."""
    assert (  # noqa: S101
        border_value is None or border_value == 0
    ), f"Here border_value is not used, and defaultly should be None or 0. got {border_value}."

    axis = 0 if direction == "horizontal" else 1
    out = out_shape[1] if direction == "horizontal" else out_shape[0]

    for polygon in polygons:
        p = np.asarray(polygon.points)
        p[axis::2] = np.clip(p[axis::2] + offset, 0, out)
        polygon.points = p.tolist()
    return polygons


def _get_translate_matrix(offset: int | float, direction: str = "horizontal") -> np.ndarray:
    """Generate the translate matrix.

    Args:
        offset (int | float): The offset used for translate.
        direction (str): The translate direction, either
            "horizontal" or "vertical".

    Returns:
        ndarray: The translate matrix with dtype float32.
    """
    if direction == "horizontal":
        translate_matrix = np.float32([[1, 0, offset], [0, 1, 0]])
    elif direction == "vertical":
        translate_matrix = np.float32([[1, 0, 0], [0, 1, offset]])
    return translate_matrix


def clip_bboxes(boxes: Tensor, img_shape: tuple[int, int]) -> Tensor:
    """Clip boxes according to the image shape in-place.

    Args:
        img_shape (tuple[int, int]): A tuple of image height and width.

    Returns:
        (Tensor): Clipped boxes.
    """
    h, w = img_shape
    boxes[..., 0::2] = boxes[..., 0::2].clamp(0, w)
    boxes[..., 1::2] = boxes[..., 1::2].clamp(0, h)
    return boxes


def is_inside_bboxes(
    boxes: Tensor,
    img_shape: tuple[int, int],
    all_inside: bool = False,
    allowed_border: int = 0,
) -> BoolTensor:
    """Find boxes inside the image.

    Args:
        boxes (Tensor): Bounding boxes to be checked.
        img_shape (tuple[int, int]): A tuple of image height and width.
        all_inside (bool): Whether the boxes are all inside the image or
            part inside the image. Defaults to False.
        allowed_border (int): Boxes that extend beyond the image shape
            boundary by more than ``allowed_border`` are considered
            "outside" Defaults to 0.

    Returns:
        (BoolTensor): A BoolTensor indicating whether the box is inside
            the image. Assuming the original boxes have shape (m, n, 4),
            the output has shape (m, n).
    """
    img_h, img_w = img_shape
    if all_inside:
        return (
            (boxes[:, 0] >= -allowed_border)
            & (boxes[:, 1] >= -allowed_border)
            & (boxes[:, 2] < img_w + allowed_border)
            & (boxes[:, 3] < img_h + allowed_border)
        )
    return (
        (boxes[..., 0] < img_w + allowed_border)
        & (boxes[..., 1] < img_h + allowed_border)
        & (boxes[..., 2] > -allowed_border)
        & (boxes[..., 3] > -allowed_border)
    )


def flip_bboxes(boxes: Tensor, img_shape: tuple[int, int], direction: str = "horizontal") -> Tensor:
    """Flip boxes horizontally or vertically in-place.

    Args:
        boxes (Tensor): Bounding boxes to be flipped.
        img_shape (Tuple[int, int]): A tuple of image height and width.
        direction (str): Flip direction, options are "horizontal",
            "vertical" and "diagonal". Defaults to "horizontal"

    Returns:
        (Tensor): Flipped bounding boxes.
    """
    assert direction in ["horizontal", "vertical", "diagonal"]  # noqa: S101
    flipped = boxes.clone()
    if direction == "horizontal":
        flipped[..., 0] = img_shape[1] - boxes[..., 2]
        flipped[..., 2] = img_shape[1] - boxes[..., 0]
    elif direction == "vertical":
        flipped[..., 1] = img_shape[0] - boxes[..., 3]
        flipped[..., 3] = img_shape[0] - boxes[..., 1]
    else:
        flipped[..., 0] = img_shape[1] - boxes[..., 2]
        flipped[..., 1] = img_shape[0] - boxes[..., 3]
        flipped[..., 2] = img_shape[1] - boxes[..., 0]
        flipped[..., 3] = img_shape[0] - boxes[..., 1]
    return flipped


def overlap_bboxes(
    bboxes1: Tensor,
    bboxes2: Tensor,
    mode: str = "iou",
    is_aligned: bool = False,
    eps: float = 1e-6,
) -> Tensor:
    """Calculate overlap between two set of bboxes.

    FP16 Contributed by https://github.com/open-mmlab/mmdetection/pull/4889
    Note:
        Assume bboxes1 is M x 4, bboxes2 is N x 4, when mode is 'iou',
        there are some new generated variable when calculating IOU
        using overlap_bboxes function:

        1) is_aligned is False
            area1: M x 1
            area2: N x 1
            lt: M x N x 2
            rb: M x N x 2
            wh: M x N x 2
            overlap: M x N x 1
            union: M x N x 1
            ious: M x N x 1

            Total memory:
                S = (9 x N x M + N + M) * 4 Byte,

            When using FP16, we can reduce:
                R = (9 x N x M + N + M) * 4 / 2 Byte
                R large than (N + M) * 4 * 2 is always true when N and M >= 1.
                Obviously, N + M <= N * M < 3 * N * M, when N >=2 and M >=2,
                           N + 1 < 3 * N, when N or M is 1.

            Given M = 40 (ground truth), N = 400000 (three anchor boxes
            in per grid, FPN, R-CNNs),
                R = 275 MB (one times)

            A special case (dense detection), M = 512 (ground truth),
                R = 3516 MB = 3.43 GB

            When the batch size is B, reduce:
                B x R

            Therefore, CUDA memory runs out frequently.

            Experiments on GeForce RTX 2080Ti (11019 MiB):

            |   dtype   |   M   |   N   |   Use    |   Real   |   Ideal   |
            |:----:|:----:|:----:|:----:|:----:|:----:|
            |   FP32   |   512 | 400000 | 8020 MiB |   --   |   --   |
            |   FP16   |   512 | 400000 |   4504 MiB | 3516 MiB | 3516 MiB |
            |   FP32   |   40 | 400000 |   1540 MiB |   --   |   --   |
            |   FP16   |   40 | 400000 |   1264 MiB |   276MiB   | 275 MiB |

        2) is_aligned is True
            area1: N x 1
            area2: N x 1
            lt: N x 2
            rb: N x 2
            wh: N x 2
            overlap: N x 1
            union: N x 1
            ious: N x 1

            Total memory:
                S = 11 x N * 4 Byte

            When using FP16, we can reduce:
                R = 11 x N * 4 / 2 Byte

        So do the 'giou' (large than 'iou').

        Time-wise, FP16 is generally faster than FP32.

        When gpu_assign_thr is not -1, it takes more time on cpu
        but not reduce memory.
        There, we can reduce half the memory and keep the speed.

    If ``is_aligned`` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned`` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = overlap_bboxes(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = overlap_bboxes(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )

    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(overlap_bboxes(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(overlap_bboxes(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(overlap_bboxes(empty, empty).shape) == (0, 0)
    """
    assert mode in ["iou", "iof", "giou"], f"Unsupported mode {mode}"  # noqa: S101
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert bboxes1.size(-1) == 4 or bboxes1.size(0) == 0  # noqa: S101
    assert bboxes2.size(-1) == 4 or bboxes2.size(0) == 0  # noqa: S101

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]  # noqa: S101
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols  # noqa: S101

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new((*batch_shape, rows))
        return bboxes1.new((*batch_shape, rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        union = area1 + area2 - overlap if mode in ["iou", "giou"] else area1
        if mode == "giou":
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        union = area1[..., None] + area2[..., None, :] - overlap if mode in ["iou", "giou"] else area1[..., None]
        if mode == "giou":
            enclosed_lt = torch.min(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ["iou", "iof"]:
        return ious
    # calculate gious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    return ious - (enclose_area - union) / enclose_area


def centers_bboxes(boxes: Tensor) -> Tensor:
    """Return a tensor representing the centers of boxes."""
    return (boxes[..., :2] + boxes[..., 2:]) / 2


def fp16_clamp(x: Tensor, min: float | None = None, max: float | None = None) -> Tensor:  # noqa: A002
    """Clamp fp16 tensor."""
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)


def scale_size(
    size: tuple[int, int],
    scale: float | int | tuple[float, float] | tuple[int, int],
) -> tuple[int, int]:
    """Rescale a size by a ratio.

    Args:
        size (tuple[int]): (height, width).
        scale (float | int | tuple(float) | tuple(int)): Scaling factor with (height, width).

    Returns:
        tuple[int]: scaled size with (height, width).
    """
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    h, w = size
    return int(h * float(scale[0]) + 0.5), int(w * float(scale[1]) + 0.5)


def rescale_size(
    old_size: tuple,
    scale: float | int | tuple[float, float] | tuple[int, int],
    return_scale: bool = False,
) -> tuple[int, int] | tuple[tuple[int, int], float | int]:
    """Calculate the new size to be rescaled to.

    Args:
        old_size (tuple[int]): The old size (height, width) of image.
        scale (float | int | tuple[float] | tuple[int]): The scaling factor or maximum size.
            If it is a float number, an integer, or a tuple of 2 float numbers,
            then the image will be rescaled by this factor, else if it is a tuple of 2 integers,
            then the image will be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the rescaled image size.

    Returns:
        tuple[int]: The new rescaled image size with (height, width).
            If return_scale is True, scale_factor obtained again will be returned as well.
    """
    h, w = old_size
    msg = ""
    if isinstance(scale, (float, int)):
        if scale <= 0:
            msg = f"Invalid scale {scale}, must be positive."
            raise ValueError(msg)
        scale_factor = scale
    elif isinstance(scale, tuple):
        if isinstance(scale[0], int):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
        elif isinstance(scale[0], float):
            scale_factor = scale  # type: ignore[assignment]
        else:
            msg = f"Scale must be a number or tuple of int/float, but got tuple of {type(scale[0])}"
    else:
        msg = f"Scale must be a number or tuple of int/float, but got {type(scale)}"

    if msg:
        raise TypeError(msg)

    new_size = scale_size((h, w), scale_factor)

    if return_scale:
        return new_size, scale_factor
    return new_size


def flip_image(img: np.ndarray | list[np.ndarray], direction: str = "horizontal") -> np.ndarray | list[np.ndarray]:
    """Flip an image horizontally or vertically.

    Args:
        img (ndarray): Image to be flipped.
        direction (str): The flip direction, either "horizontal" or
            "vertical" or "diagonal".

    Returns:
        ndarray: The flipped image.
    """
    if direction not in ["horizontal", "vertical", "diagonal"]:
        msg = f"direction (={direction}) should be in one of ('horizontal', 'vertical', 'diagonal')."
        raise ValueError(msg)

    if isinstance(img, list):
        return [flip_image(im, direction) for im in img]

    if direction == "horizontal":
        return np.flip(img, axis=1)
    elif direction == "vertical":  # noqa: RET505
        return np.flip(img, axis=0)
    else:
        return np.flip(img, axis=(0, 1))


def flip_masks(masks: np.ndarray, direction: str = "horizontal") -> np.ndarray:
    """Flip masks alone the given direction."""
    assert direction in ("horizontal", "vertical", "diagonal")  # noqa: S101

    return np.stack([flip_image(mask, direction=direction) for mask in masks])


def flip_polygons(polygons: list[Polygon], height: int, width: int, direction: str = "horizontal") -> list[Polygon]:
    """Flip polygons alone the given direction."""
    for polygon in polygons:
        p = np.asarray(polygon.points)
        if direction == "horizontal":
            p[0::2] = width - p[0::2]
        elif direction == "vertical":
            p[1::2] = height - p[1::2]
        else:
            p[0::2] = width - p[0::2]
            p[1::2] = height - p[1::2]
        polygon.points = p.tolist()
    return polygons


def project_bboxes(boxes: Tensor, homography_matrix: Tensor | np.ndarray) -> Tensor:
    """Geometric transformat boxes in-place.

    Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/structures/bbox/horizontal_boxes.py#L184-L202

    Args:
        homography_matrix (Tensor or np.ndarray]):
            Shape (3, 3) for geometric transformation.

    Returns:
        (Tensor | np.ndarray): Projected bounding boxes.
    """
    if isinstance(homography_matrix, np.ndarray):
        homography_matrix = boxes.new_tensor(homography_matrix)
    corners = hbox2corner(boxes)
    corners = torch.cat([corners, corners.new_ones(*corners.shape[:-1], 1)], dim=-1)
    corners_t = torch.transpose(corners, -1, -2)
    corners_t = torch.matmul(homography_matrix, corners_t)
    corners = torch.transpose(corners_t, -1, -2)
    # Convert to homogeneous coordinates by normalization
    corners = corners[..., :2] / corners[..., 2:3]
    return corner2hbox(corners)


def hbox2corner(boxes: Tensor) -> Tensor:
    """Convert box coordinates from (x1, y1, x2, y2) to corners ((x1, y1), (x2, y1), (x1, y2), (x2, y2)).

    Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/structures/bbox/horizontal_boxes.py#L204-L217

    Args:
        boxes (Tensor): Horizontal box tensor with shape of (..., 4).

    Returns:
        Tensor: Corner tensor with shape of (..., 4, 2).
    """
    x1, y1, x2, y2 = torch.split(boxes, 1, dim=-1)
    corners = torch.cat([x1, y1, x2, y1, x1, y2, x2, y2], dim=-1)
    return corners.reshape(*corners.shape[:-1], 4, 2)


def corner2hbox(corners: Tensor) -> Tensor:
    """Convert box coordinates from corners ((x1, y1), (x2, y1), (x1, y2), (x2, y2)) to (x1, y1, x2, y2).

    Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/structures/bbox/horizontal_boxes.py#L219-L234

    Args:
        corners (Tensor): Corner tensor with shape of (..., 4, 2).

    Returns:
        Tensor: Horizontal box tensor with shape of (..., 4).
    """
    if corners.numel() == 0:
        return corners.new_zeros((0, 4))
    min_xy = corners.min(dim=-2)[0]
    max_xy = corners.max(dim=-2)[0]
    return torch.cat([min_xy, max_xy], dim=-1)


def crop_masks(masks: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Crop each mask by the given bbox."""
    assert isinstance(bbox, np.ndarray)  # noqa: S101
    assert bbox.ndim == 1  # noqa: S101

    height, width = masks.shape[1:]

    # clip the boundary
    bbox = bbox.copy()
    bbox[0::2] = np.clip(bbox[0::2], 0, width)
    bbox[1::2] = np.clip(bbox[1::2], 0, height)
    x1, y1, x2, y2 = bbox
    w = np.maximum(x2 - x1, 1)
    h = np.maximum(y2 - y1, 1)

    return masks[:, y1 : y1 + h, x1 : x1 + w]


def crop_polygons(polygons: list[Polygon], bbox: np.ndarray, height: int, width: int) -> list[Polygon]:
    """Crop each polygon by the given bbox."""
    assert isinstance(bbox, np.ndarray)  # noqa: S101
    assert bbox.ndim == 1  # noqa: S101

    # clip the boundary
    bbox = bbox.copy()
    bbox[0::2] = np.clip(bbox[0::2], 0, width)
    bbox[1::2] = np.clip(bbox[1::2], 0, height)
    x1, y1, x2, y2 = bbox

    # reference: https://github.com/facebookresearch/fvcore/blob/main/fvcore/transforms/transform.py
    crop_box = geometry.box(x1, y1, x2, y2).buffer(0.0)
    # suppress shapely warnings util it incorporates GEOS>=3.11.2
    # reference: https://github.com/shapely/shapely/issues/1345
    initial_settings = np.seterr()
    np.seterr(invalid="ignore")
    for polygon in polygons:
        cropped_poly_per_obj: list[Polygon] = []

        p = np.asarray(polygon.points).copy()
        p = geometry.Polygon(p.reshape(-1, 2)).buffer(0.0)
        # polygon must be valid to perform intersection.
        if not p.is_valid:
            # a dummy polygon to avoid misalignment between masks and boxes
            polygon.points = [0, 0, 0, 0, 0, 0]
            continue

        cropped = p.intersection(crop_box)
        if cropped.is_empty:
            # a dummy polygon to avoid misalignment between masks and boxes
            polygon.points = [0, 0, 0, 0, 0, 0]
            continue

        cropped = cropped.geoms if isinstance(cropped, geometry.collection.BaseMultipartGeometry) else [cropped]

        # one polygon may be cropped to multiple ones
        for poly in cropped:
            # ignore lines or points
            if not isinstance(poly, geometry.Polygon) or not poly.is_valid:
                continue

            coords = np.asarray(poly.exterior.coords)

            # remove an extra identical vertex at the end
            coords = coords[:-1]
            coords[:, 0] -= x1
            coords[:, 1] -= y1
            cropped_poly_per_obj.append(coords.reshape(-1).tolist())

        # a dummy polygon to avoid misalignment between masks and boxes
        if len(cropped_poly_per_obj) == 0:
            cropped_poly_per_obj.append([0, 0, 0, 0, 0, 0])

        polygon.points = list(itertools.chain(*cropped_poly_per_obj))
    np.seterr(**initial_settings)
    return polygons


def get_bboxes_from_masks(masks: Tensor) -> np.ndarray:
    """Create boxes from masks."""
    num_masks = len(masks)
    bboxes = np.zeros((num_masks, 4), dtype=np.float32)

    x_any = masks.any(axis=1)
    y_any = masks.any(axis=2)
    for idx in range(num_masks):
        x = np.where(x_any[idx, :])[0]
        y = np.where(y_any[idx, :])[0]
        if len(x) > 0 and len(y) > 0:
            # use +1 for x_max and y_max so that the right and bottom
            # boundary of instance masks are fully included by the box
            bboxes[idx, :] = np.array([x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=np.float32)
    return bboxes


def get_bboxes_from_polygons(polygons: list[Polygon], height: int, width: int) -> np.ndarray:
    """Create boxes from polygons."""
    num_polygons = len(polygons)
    boxes = np.zeros((num_polygons, 4), dtype=np.float32)
    for idx, polygon in enumerate(polygons):
        # simply use a number that is big enough for comparison with coordinates
        xy_min = np.array([width * 2, height * 2], dtype=np.float32)
        xy_max = np.zeros(2, dtype=np.float32)

        xy = np.array(polygon.points).reshape(-1, 2).astype(np.float32)
        xy_min = np.minimum(xy_min, np.min(xy, axis=0))
        xy_max = np.maximum(xy_max, np.max(xy, axis=0))
        boxes[idx, :2] = xy_min
        boxes[idx, 2:] = xy_max
    return boxes


def area_polygon(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute the area of a component of a polygon.

    Using the shoelace formula:
    https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates

    Args:
        x (ndarray): x coordinates of the component
        y (ndarray): y coordinates of the component

    Return:
        (float): the are of the component
    """
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
