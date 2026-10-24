# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Torchvision kernel registrations for getitune data entities.

This module registers torchvision v2 functional kernels for getitune-specific
tensor types (``ImageInfo``), enabling these types to be processed by
standard torchvision transforms.
"""

from __future__ import annotations

import torchvision.transforms.v2.functional as F  # noqa: N812

from getitune.data.entity.base import ImageInfo


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
    """Register ImageInfo to TorchVision v2 crop kernel."""
    image_info.img_shape = (height, width)
    image_info.scale_factor = None
    return image_info


@F.register_kernel(functional=F.resized_crop, tv_tensor_cls=ImageInfo)
def _resized_crop_image_info(
    image_info: ImageInfo,
    size: list[int],
    **kwargs,  # noqa: ARG001
) -> ImageInfo:
    """Register ImageInfo to TorchVision v2 resized_crop kernel."""
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
    """Register ImageInfo to TorchVision v2 center_crop kernel."""
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
    """Register ImageInfo to TorchVision v2 pad kernel."""
    left, right, top, bottom = F._geometry._parse_pad_padding(padding)  # noqa: SLF001
    height, width = image_info.img_shape
    image_info.padding = (left, top, right, bottom)
    image_info.img_shape = (height + top + bottom, width + left + right)
    return image_info
