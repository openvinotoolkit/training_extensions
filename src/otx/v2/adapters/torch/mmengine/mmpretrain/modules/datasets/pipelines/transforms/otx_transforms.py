"""Module for defining transforms used for OTX classification."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import random

from mmpretrain.datasets.transforms import TRANSFORMS, PackInputs
from PIL import Image
from torchvision.transforms import functional


@TRANSFORMS.register_module()
class PILToTensor:
    """Convert PIL image to Tensor."""

    def __call__(self, results):
        """Call function of PILToTensor class."""
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            if not Image.isImageType(img):
                img = Image.fromarray(img)
            img = functional.to_tensor(img)
            results["PILToTensor"] = True
            results[key] = img
        return results


@TRANSFORMS.register_module()
class TensorNormalize:
    """Normalize tensor object."""

    def __init__(self, mean, std, inplace=False) -> None:
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, results):
        """Call function of TensorNormalize class."""
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            img = functional.normalize(img, self.mean, self.std, self.inplace)
            results["TensorNormalize"] = True
            results[key] = img
        return results


# TODO [Jihwan]: Can be removed by mmpretrain.dataset.pipelines.auto_augment L398, Roate class
@TRANSFORMS.register_module()
class RandomRotate:
    """Random rotate, from torchreid.data.transforms."""

    def __init__(self, p=0.5, angle=(-5, 5), values=None) -> None:
        self.p = p
        self.angle = angle
        self.discrete = values is not None and len([v for v in values if v != 0]) > 0
        self.values = values

    def __call__(self, results, *args, **kwargs):
        """Call function of RandomRotate class."""
        # disable B311 random - used for the random sampling not for security/crypto
        if random.uniform(0, 1) > self.p:  # nosec B311
            return results
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            if self.discrete:
                # disable B311 random - used for the random sampling not for security/crypto
                rnd_angle = float(self.values[random.randint(0, len(self.values) - 1)])  # nosec B311
            else:
                # disable B311 random - used for the random sampling not for security/crypto
                rnd_angle = random.randint(self.angle[0], self.angle[1])  # nosec B311
            if not Image.isImageType(img):
                img = Image.fromarray(img)

            img = functional.rotate(img, rnd_angle, expand=False, center=None)
            results["RandomRotate"] = True
            results[key] = img

        return results


@TRANSFORMS.register_module()
class PackMultiKeyInputs(PackInputs):
    def __init__(self, multi_key=[], **kwargs) -> None:
        super().__init__(**kwargs)
        self.multi_key = multi_key

    def transform(self, results: dict) -> dict:
        """Method to pack the input data."""

        packed_results = super().transform(results=results)
        for key in self.multi_key:
            if key in results:
                input_ = results[key]
                packed_results[key] = self.format_input(input_)
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(input_key='{self.multi_key}', "
        repr_str += f"algorithm_keys={self.algorithm_keys}, "
        repr_str += f"meta_keys={self.meta_keys})"
        return repr_str
