# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

import copy
import numpy as np
import functools
import inspect
import weakref
import torch
from torch import Tensor, BoolTensor


class cache_randomness:
    """Decorator that marks the method with random return value(s) in a
    transform class.

    Reference : https://github.com/open-mmlab/mmcv/blob/v2.1.0/mmcv/transforms/utils.py#L15-L87

    This decorator is usually used together with the context-manager
    :func`:cache_random_params`. In this context, a decorated method will
    cache its return value(s) at the first time of being invoked, and always
    return the cached values when being invoked again.

    .. note::
        Only an instance method can be decorated with ``cache_randomness``.
    """

    def __init__(self, func):

        # Check `func` is to be bound as an instance method
        if not inspect.isfunction(func):
            raise TypeError('Unsupport callable to decorate with'
                            '@cache_randomness.')
        func_args = inspect.getfullargspec(func).args
        if len(func_args) == 0 or func_args[0] != 'self':
            raise TypeError(
                '@cache_randomness should only be used to decorate '
                'instance methods (the first argument is ``self``).')

        functools.update_wrapper(self, func)
        self.func = func
        self.instance_ref = None

    def __set_name__(self, owner, name):
        # Maintain a record of decorated methods in the class
        if not hasattr(owner, '_methods_with_randomness'):
            setattr(owner, '_methods_with_randomness', [])

        # Here `name` equals to `self.__name__`, i.e., the name of the
        # decorated function, due to the invocation of `update_wrapper` in
        # `self.__init__()`
        owner._methods_with_randomness.append(name)

    def __call__(self, *args, **kwargs):
        # Get the transform instance whose method is decorated
        # by cache_randomness
        instance = self.instance_ref()
        name = self.__name__

        # Check the flag ``self._cache_enabled``, which should be
        # set by the contextmanagers like ``cache_random_parameters```
        cache_enabled = getattr(instance, '_cache_enabled', False)

        if cache_enabled:
            # Initialize the cache of the transform instances. The flag
            # ``cache_enabled``` is set by contextmanagers like
            # ``cache_random_params```.
            if not hasattr(instance, '_cache'):
                setattr(instance, '_cache', {})

            if name not in instance._cache:
                instance._cache[name] = self.func(instance, *args, **kwargs)
            # Return the cached value
            return instance._cache[name]
        else:
            # Clear cache
            if hasattr(instance, '_cache'):
                del instance._cache
            # Return function output
            return self.func(instance, *args, **kwargs)

    def __get__(self, obj, cls):
        self.instance_ref = weakref.ref(obj)
        # Return a copy to avoid multiple transform instances sharing
        # one `cache_randomness` instance, which may cause data races
        # in multithreading cases.
        return copy.copy(self)


def to_np_image(img: Tensor) -> np.ndarray:
    """Convert torch.Tensor 3D image to numpy 3D image.

    TODO (sungchul): move it into base data entity?
    
    """
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
        scale_factor (tuple[float, float]): factors for scaling boxes.
            The length should be 2.

    Returns:
        (Tensor): rescaled bounding boxes.
    """
    assert len(scale_factor) == 2
    scale_factor = boxes.new_tensor(scale_factor).repeat(2)
    return boxes * scale_factor


def translate_bboxes(boxes: Tensor, distances: tuple[float, float]) -> Tensor:
    """Translate boxes in-place.

    Args:
        boxes (Tensor): Bounding boxes to be translated.
        distances (tuple[float, float]): Translate distances. The first
            is horizontal distance and the second is vertical distance.
    
    Returns:
        (Tensor): Translated bounding boxes.
    """
    assert len(distances) == 2
    return boxes + boxes.new_tensor(distances).repeat(2)


def clip_bboxes(boxes: Tensor, img_shape: tuple[int, int]) -> Tensor:
    """Clip boxes according to the image shape in-place.

    Args:
        img_shape (tuple[int, int]): A tuple of image height and width.
    """
    boxes[..., 0::2] = boxes[..., 0::2].clamp(0, img_shape[1])
    boxes[..., 1::2] = boxes[..., 1::2].clamp(0, img_shape[0])
    return boxes


def is_inside_bboxes(boxes: Tensor, img_shape: tuple[int, int], all_inside: bool = False, allowed_border: int = 0) -> BoolTensor:
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
        return (boxes[:, 0] >= -allowed_border) & \
            (boxes[:, 1] >= -allowed_border) & \
            (boxes[:, 2] < img_w + allowed_border) & \
            (boxes[:, 3] < img_h + allowed_border)
    else:
        return (boxes[..., 0] < img_w + allowed_border) & \
            (boxes[..., 1] < img_h + allowed_border) & \
            (boxes[..., 2] > -allowed_border) & \
            (boxes[..., 3] > -allowed_border)


def flip_bboxes(boxes: Tensor, img_shape: tuple[int, int], direction: str = 'horizontal') -> Tensor:
        """Flip boxes horizontally or vertically in-place.

        Args:
            boxes (Tensor): Bounding boxes to be flipped.
            img_shape (Tuple[int, int]): A tuple of image height and width.
            direction (str): Flip direction, options are "horizontal",
                "vertical" and "diagonal". Defaults to "horizontal"

        Returns:
            (Tensor): Flipped bounding boxes.
        """
        assert direction in ['horizontal', 'vertical', 'diagonal']
        flipped = boxes.clone()
        if direction == 'horizontal':
            flipped[..., 0] = img_shape[1] - boxes[..., 2]
            flipped[..., 2] = img_shape[1] - boxes[..., 0]
        elif direction == 'vertical':
            flipped[..., 1] = img_shape[0] - boxes[..., 3]
            flipped[..., 3] = img_shape[0] - boxes[..., 1]
        else:
            flipped[..., 0] = img_shape[1] - boxes[..., 2]
            flipped[..., 1] = img_shape[0] - boxes[..., 3]
            flipped[..., 2] = img_shape[1] - boxes[..., 0]
            flipped[..., 3] = img_shape[0] - boxes[..., 1]
        return flipped


def overlap_bboxes(bboxes1: Tensor, bboxes2: Tensor, mode='iou', is_aligned=False, eps=1e-6):
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

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


def centers_bboxes(boxes: Tensor) -> Tensor:
    """Return a tensor representing the centers of boxes."""
    return (boxes[..., :2] + boxes[..., 2:]) / 2


def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)


def _scale_size(
    size: tuple[int, int],
    scale: float | int | tuple[float, float] | tuple[int, int],
) -> tuple[int, int]:
    """Rescale a size by a ratio.

    Args:
        size (tuple[int]): (w, h).
        scale (float | int | tuple(float) | tuple(int)): Scaling factor.

    Returns:
        tuple[int]: scaled size.
    """
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    w, h = size
    return int(w * float(scale[0]) + 0.5), int(h * float(scale[1]) + 0.5)


def rescale_size(old_size: tuple,
                 scale: float | int | tuple[int, int],
                 return_scale: bool = False) -> tuple:
    """Calculate the new size to be rescaled to.

    Args:
        old_size (tuple[int]): The old size (w, h) of image.
        scale (float | int | tuple[int]): The scaling factor or maximum size.
            If it is a float number or an integer, then the image will be
            rescaled by this factor, else if it is a tuple of 2 integers, then
            the image will be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image size.

    Returns:
        tuple[int]: The new rescaled image size.
    """
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f'Invalid scale {scale}, must be positive.')
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
    else:
        raise TypeError(
            f'Scale must be a number or tuple of int, but got {type(scale)}')

    new_size = _scale_size((w, h), scale_factor)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size


def flip_image(img: np.ndarray, direction: str = 'horizontal') -> np.ndarray:
    """Flip an image horizontally or vertically.

    Args:
        img (ndarray): Image to be flipped.
        direction (str): The flip direction, either "horizontal" or
            "vertical" or "diagonal".

    Returns:
        ndarray: The flipped image.
    """
    assert direction in ['horizontal', 'vertical', 'diagonal']
    if direction == 'horizontal':
        return np.flip(img, axis=1)
    elif direction == 'vertical':
        return np.flip(img, axis=0)
    else:
        return np.flip(img, axis=(0, 1))


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
    corners = torch.cat(
        [corners, corners.new_ones(*corners.shape[:-1], 1)], dim=-1)
    corners_T = torch.transpose(corners, -1, -2)
    corners_T = torch.matmul(homography_matrix, corners_T)
    corners = torch.transpose(corners_T, -1, -2)
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
