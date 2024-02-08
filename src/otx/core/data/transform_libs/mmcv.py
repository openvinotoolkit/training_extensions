# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Helper to support MMCV data transform functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import cv2
import numpy as np
import mmcv
from mmcv.transforms import LoadImageFromFile as MMCVLoadImageFromFile
from mmcv.transforms import Resize as MMCVResize
from mmcv.transforms.builder import TRANSFORMS
from openvino.model_api.adapters.utils import resize_image_with_aspect_ocv, resize_image_letterbox_ocv

from otx.core.data.entity.base import OTXDataEntity
from otx.core.utils.config import convert_conf_to_mmconfig_dict

if TYPE_CHECKING:
    from mmengine.registry import Registry

    from otx.core.config.data import SubsetConfig


@TRANSFORMS.register_module(force=True)
class LoadImageFromFile(MMCVLoadImageFromFile):
    """Class to override MMCV LoadImageFromFile."""

    def transform(self, entity: OTXDataEntity) -> dict | None:
        """Transform OTXDataEntity to MMCV data entity format."""
        img: np.ndarray = entity.image

        if self.to_float32:
            img = img.astype(np.float32)

        results = {}
        results["img"] = img
        results["img_shape"] = img.shape[:2]
        results["ori_shape"] = img.shape[:2]

        results["__otx__"] = entity

        return results


@TRANSFORMS.register_module()
class ResizeWithAspectMAPI(MMCVResize):
    """Resize that reproduces `fit_to_window` in MAPI."""
    def __init__(self, scale: tuple[int, int], **kwargs) -> None:
        if not isinstance(scale, tuple):
            raise RuntimeError
        super().__init__(keep_ratio=True, scale=scale, **kwargs)
        self.scale = scale

    def _resize_img(self, results: dict) -> None:
        """Resize images with ``results['scale']``."""

        if results.get('img', None) is not None:
            if self.keep_ratio:
                #img = resize_image_letterbox_ocv(results['img'], self.scale)
                img = resize_image_with_aspect_ocv(results['img'], self.scale)
                h, w = results['img'].shape[:2]
                nh, nw = img.shape[:2]
                scale = min(self.scale[1] / h, self.scale[0] / w)
                w_scale, h_scale = scale, scale
                dw = (w - nw) // 2
                dh = (h - nh) // 2
                """
                scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))

                results['scale'],
                w, h = size
                return int(w * float(scale[0]) + 0.5), int(h * float(scale[1]) + 0.5)

                img, scale_factor = mmcv.imrescale(
                    results['img'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results['img'].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
                """
                print(img.shape)
            else:
                raise RuntimeError

            results['img'] = img
            results['img_shape'] = img.shape[:2]
            results['scale_factor'] = (w_scale, h_scale)
            results['keep_ratio'] = self.keep_ratio
            #results['kp_shift'] = (dw, dh)

    def _resize_segg(self, results: dict) -> None:
        """Resize semantic segmentation map with ``results['scale']``."""
        if results.get('gt_seg_map', None) is not None:
            exit(0)
            if self.keep_ratio:
                gt_seg = resize_image_letterbox_ocv(results['gt_seg_map'], self.scale, cv2.INTER_NEAREST)
            else:
                raise RuntimeError

            results['gt_seg_map'] = gt_seg

    def _resize_keypointss(self, results: dict) -> None:
        """Resize keypoints with ``results['scale_factor']``."""
        if results.get('gt_keypoints', None) is not None:
            exit(0)
            keypoints = results['gt_keypoints']

            keypoints[:, :, :2] = keypoints[:, :, :2] * np.array(
                results['scale_factor']) + np.array(results['kp_shift'])
            if self.clip_object_border:
                keypoints[:, :, 0] = np.clip(keypoints[:, :, 0], 0,
                                             results['img_shape'][1])
                keypoints[:, :, 1] = np.clip(keypoints[:, :, 1], 0,
                                             results['img_shape'][0])
            results['gt_keypoints'] = keypoints

    def _resize_bboxess(self, results: dict) -> None:
        """Resize bounding boxes with ``results['scale_factor']``."""
        if results.get('gt_bboxes', None) is not None:
            bboxes = results['gt_bboxes'] * np.tile(
                np.array(results['scale_factor']), 2)
            print(bboxes)# + np.tile(np.array(results['kp_shift']), 2)
            exit(0)
            if self.clip_object_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0,
                                          results['img_shape'][1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0,
                                          results['img_shape'][0])
            results['gt_bboxes'] = bboxes


class MMCVTransformLib:
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
            mandatory_transforms={LoadImageFromFile},
        )

        return transforms
