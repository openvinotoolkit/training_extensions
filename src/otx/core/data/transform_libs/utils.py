# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

import copy
import functools
import inspect
import weakref
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
