"""Torchvision transforms to MMDetection pipeline."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
from mmcv.utils import build_from_cfg
from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines.formatting import ImageToTensor, to_tensor
from PIL import Image, ImageFilter
from torchvision import transforms as T


@PIPELINES.register_module()
class ColorJitter(T.ColorJitter):
    """MMDet adapter."""

    def __init__(self, key_maps=None, **kwargs):
        super().__init__(**kwargs)
        key_maps = key_maps if key_maps else list([("img", "img")])
        self.key_maps = key_maps

    def forward(self, img):
        """Forward function of ColorJitter."""
        outputs = img.copy()
        for key_map in self.key_maps:
            outputs[key_map[0]] = super().forward(img[key_map[1]])
        return outputs


@PIPELINES.register_module()
class RandomGrayscale(T.RandomGrayscale):
    """MMDet adapter."""

    def __init__(self, key_maps=None, **kwargs):
        super().__init__(**kwargs)
        key_maps = key_maps if key_maps else list([("img", "img")])
        self.key_maps = key_maps

    def forward(self, img):
        """Forward function of RandomGrayscale."""
        outputs = img.copy()
        for key_map in self.key_maps:
            outputs[key_map[0]] = super().forward(img[key_map[1]])
        return outputs


@PIPELINES.register_module()
class RandomErasing(T.RandomErasing):
    """MMDet adapter."""

    def __init__(self, key_maps=None, **kwargs):
        super().__init__(**kwargs)
        key_maps = key_maps if key_maps else list([("img", "img")])
        self.key_maps = key_maps

    def forward(self, img):
        """Forward function of RandomErasing."""
        outputs = img.copy()
        for key_map in self.key_maps:
            outputs[key_map[0]] = super().forward(img[key_map[1]])
        return outputs


@PIPELINES.register_module()
class RandomGaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709."""

    def __init__(self, sigma_min, sigma_max, key_maps=None):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        self.key_maps = key_maps if key_maps else list([("img", "img")])

    def __call__(self, inputs):
        """Call function of RandomGaussianBlur."""
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
        """Repr function of RandomGaussianBlur."""
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class RandomApply(T.RandomApply):
    """MMDet adapter."""

    def __init__(self, transform_cfgs, p=0.5):
        transforms = []
        for transform_cfg in transform_cfgs:
            transforms.append(build_from_cfg(transform_cfg, PIPELINES))
        super().__init__(transforms, p=p)


@PIPELINES.register_module()
class NDArrayToTensor(ImageToTensor):
    """MMDet adapter."""

    def __call__(self, results):
        """Call function of NDArrayToTensor."""
        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results[key] = to_tensor(img)
        return results


@PIPELINES.register_module()
class NDArrayToPILImage:
    """NDArrayToPILImage."""

    def __init__(self, keys=None):
        self.keys = keys if keys else list(["img"])

    def __call__(self, results):
        """Call function of NDArrayToPILImage."""
        for key in self.keys:
            img = results[key]
            img = Image.fromarray(img, mode="RGB")
            results[key] = img
        return results

    def __repr__(self):
        """Repr function of NDArrayToPILImage."""
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class PILImageToNDArray:
    """PILImageToNDArray."""

    def __init__(self, keys=None):
        self.keys = keys if keys else list(["img"])

    def __call__(self, results):
        """Call function of PILImageToNDArray."""
        for key in self.keys:
            img = results[key]
            img = np.asarray(img)
            results[key] = img
        return results

    def __repr__(self):
        """Repr function of PILImageToNDArray."""
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class BranchImage:
    """BranchImage."""

    def __init__(self, key_map=None):
        self.key_map = key_map if key_map else dict()

    def __call__(self, results):
        """Call function of BranchImage."""
        for key_1, key_2 in self.key_map.items():
            if key_1 in results:
                results[key_2] = results[key_1]
            if key_1 in results["img_fields"]:
                results["img_fields"].append(key_2)
        return results

    def __repr__(self):
        """Repr function of BranchImage."""
        repr_str = self.__class__.__name__
        return repr_str
