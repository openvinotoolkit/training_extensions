# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from mmseg.datasets import PIPELINES
from mmseg.datasets.pipelines.formatting import to_tensor


@PIPELINES.register_module(force=True)
class Normalize(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        for target in ["img", "ul_w_img", "aux_img"]:
            if target in results:
                results[target] = mmcv.imnormalize(results[target], self.mean, self.std, self.to_rgb)
        results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb=" f"{self.to_rgb})"
        return repr_str


@PIPELINES.register_module(force=True)
class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img"
    and "gt_semantic_seg". These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        for target in ["img", "ul_w_img", "aux_img"]:
            if target not in results:
                continue

            img = results[target]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)

            if len(img.shape) == 3:
                img = np.ascontiguousarray(img.transpose(2, 0, 1)).astype(np.float32)
            elif len(img.shape) == 4:
                # for selfsl or supcon
                img = np.ascontiguousarray(img.transpose(0, 3, 1, 2)).astype(np.float32)
            else:
                raise ValueError(f"img.shape={img.shape} is not supported.")

            results[target] = DC(to_tensor(img), stack=True)

        for trg_name in ["gt_semantic_seg", "gt_class_borders", "pixel_weights"]:
            if trg_name not in results:
                continue

            out_type = np.float32 if trg_name == "pixel_weights" else np.int64
            results[trg_name] = DC(to_tensor(results[trg_name][None, ...].astype(out_type)), stack=True)

        return results

    def __repr__(self):
        return self.__class__.__name__


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
