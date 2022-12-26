# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
from mmcv.utils import build_from_cfg
from mmdet.datasets import PIPELINES

# import torchvision.transforms.functional as F
from mmdet.datasets.pipelines.formating import ImageToTensor, to_tensor

# from mmdet.datasets.pipelines.transforms import Normalize
from PIL import Image, ImageFilter

# import cv2 as cv
from torchvision import transforms as T


@PIPELINES.register_module()
class ColorJitter(T.ColorJitter):
    """MMDet adapter"""

    def __init__(self, key_maps=[("img", "img")], **kwargs):
        super().__init__(**kwargs)
        self.key_maps = key_maps

    def forward(self, inputs):
        outputs = inputs.copy()
        for key_map in self.key_maps:
            outputs[key_map[0]] = super().forward(inputs[key_map[1]])
        return outputs


@PIPELINES.register_module()
class RandomGrayscale(T.RandomGrayscale):
    """MMDet adapter"""

    def __init__(self, key_maps=[("img", "img")], **kwargs):
        super().__init__(**kwargs)
        self.key_maps = key_maps

    def forward(self, inputs):
        outputs = inputs.copy()
        for key_map in self.key_maps:
            outputs[key_map[0]] = super().forward(inputs[key_map[1]])
        return outputs


@PIPELINES.register_module()
class RandomErasing(T.RandomErasing):
    """MMDet adapter"""

    def __init__(self, key_maps=[("img", "img")], **kwargs):
        super().__init__(**kwargs)
        self.key_maps = key_maps

    def forward(self, inputs):
        outputs = inputs.copy()
        for key_map in self.key_maps:
            outputs[key_map[0]] = super().forward(inputs[key_map[1]])
        return outputs


@PIPELINES.register_module()
class RandomGaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709."""

    def __init__(self, sigma_min, sigma_max, key_maps=[("img", "img")]):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.key_maps = key_maps

    def __call__(self, inputs):
        outputs = inputs.copy()
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        # ksize = 2*int(np.ceil(2.0*sigma)) + 1
        for key_map in self.key_maps:
            img = inputs[key_map[0]]
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
            # img = cv.GaussianBlur(img, ksize=(0,0), sigmaX=sigma)
            # img = F.gaussian_blur(img, ksize, [sigma, sigma])
            outputs[key_map[1]] = img
        return outputs

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class RandomApply(T.RandomApply):
    """MMDet adapter"""

    def __init__(self, transform_cfgs, p=0.5):
        transforms = []
        for transform_cfg in transform_cfgs:
            transforms.append(build_from_cfg(transform_cfg, PIPELINES))
        super().__init__(transforms, p=p)


@PIPELINES.register_module()
class NDArrayToTensor(ImageToTensor):
    """MMDet adapter"""

    def __call__(self, results):
        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results[key] = to_tensor(img)
        return results


# @PIPELINES.register_module()
# class NormalizeTensor(Normalize):
#     """MMDet adapter"""
#
#     def __call__(self, results):
#         for key in results.get('img_fields', ['img']):
#             img = results[key]
#             img = F.normalize(img.float(), self.mean, self.std)
#             if self.to_rgb:
#                 img = img[[2, 1, 0]]
#             results[key] = img
#         results['img_norm_cfg'] = dict(
#             mean=self.mean, std=self.std, to_rgb=self.to_rgb)
#         return results
#
#
# @PIPELINES.register_module()
# class PadTensor(object):
#     def __init__(self, size_divisor=32):
#         self.size_divisor = float(size_divisor)
#
#     def __call__(self, results):
#         for key in results.get('img_fields', ['img']):
#             img = results[key]
#             h = img.shape[1]
#             w = img.shape[2]
#             H = int(np.ceil(h/self.size_divisor)*self.size_divisor)
#             W = int(np.ceil(w/self.size_divisor)*self.size_divisor)
#             padding = (0, 0, W-w, H-h)
#             # print(padding)
#             img = F.pad(img, padding)
#             results[key] = img
#             results['pad_shape'] = img.shape
#         results['pad_size_divisor'] = self.size_divisor
#         return results


@PIPELINES.register_module()
class NDArrayToPILImage(object):
    def __init__(self, keys=["img"]):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            img = results[key]
            img = Image.fromarray(img, mode="RGB")
            results[key] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class PILImageToNDArray(object):
    def __init__(self, keys=["img"]):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            img = results[key]
            img = np.asarray(img)
            results[key] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class BranchImage(object):
    def __init__(self, key_map={}):
        self.key_map = key_map

    def __call__(self, results):
        for k1, k2 in self.key_map.items():
            if k1 in results:
                results[k2] = results[k1]
            if k1 in results["img_fields"]:
                results["img_fields"].append(k2)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
