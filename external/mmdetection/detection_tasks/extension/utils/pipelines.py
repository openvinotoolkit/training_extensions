# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import copy
import fcntl

from typing import Dict, Any, Optional
import numpy as np
import os
import shutil

from ote_sdk.entities.label import Domain
from ote_sdk.utils.argument_checks import check_input_parameters_type

from mmdet.datasets.builder import PIPELINES

from ..datasets import get_annotation_mmdet_format


# Clean up cache directory per process launch
_CACHE_DIR = "/tmp/img-cache"
shutil.rmtree(_CACHE_DIR, ignore_errors=True)
os.makedirs(_CACHE_DIR)


@PIPELINES.register_module()
class LoadImageFromOTEDataset:
    """
    Pipeline element that loads an image from a OTE Dataset on the fly. Can do conversion to float 32 if needed.

    Expected entries in the 'results' dict that should be passed to this pipeline element are:
        results['dataset_item']: dataset_item from which to load the image
        results['dataset_id']: id of the dataset to which the item belongs
        results['index']: index of the item in the dataset

    :param to_float32: optional bool, True to convert images to fp32. defaults to False
    """

    @check_input_parameters_type()
    def __init__(self, to_float32: bool = False):
        self.to_float32 = to_float32
        self._pid = os.getpid()

    @staticmethod
    def _is_video_frame(media):
        # return "VideoFrame" in repr(media)
        return "Image" in repr(media)  # Uncomment for test

    def _get_cached_image(self, results: Dict[str, Any]):
        if self._is_video_frame(results["dataset_item"].media):
            subset = results["dataset_item"].subset
            index = results["index"]
            filename = os.path.join(_CACHE_DIR, f"{self._pid}-{subset}-{index:06d}.npy")
            if os.path.exists(filename):
                # Might be slower than dict key checking, but persitent
                # FIXME: faster cache checking?
                print(f"Loading cache {filename}")

                # ORIGIN
                # return np.load(filename)

                # LOCK
                with open(filename, "rb") as f:
                    fcntl.flock(f, fcntl.LOCK_SH)
                    cached_img = np.load(f)
                    fcntl.flock(f, fcntl.LOCK_UN)
                    return cached_img

        img = results["dataset_item"].numpy  # this takes long for VideoFrame
        if self.to_float32:
            img = img.astype(np.float32)

        if self._is_video_frame(results["dataset_item"].media):
            print(f"Saving cache {filename}")

            # ORIGIN
            # np.save(filename, img)

            # LOCK
            with open(filename, "wb") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                np.save(f, img)
                fcntl.flock(f, fcntl.LOCK_UN)

        return img

    @check_input_parameters_type()
    def __call__(self, results: Dict[str, Any]):
        # Get image (possibly from cache)
        img = self._get_cached_image(results)
        shape = img.shape

        assert shape[0] == results["height"], f"{shape[0]} != {results['height']}"
        assert shape[1] == results["width"], f"{shape[1]} != {results['width']}"

        # Fill outputs
        filename = f"Dataset item index {results['index']}"
        results["filename"] = filename
        results["ori_filename"] = filename
        results["img"] = img
        results["img_shape"] = shape
        results["ori_shape"] = shape
        # Set initial values for default meta_keys
        results["pad_shape"] = shape
        num_channels = 1 if len(shape) < 3 else shape[2]
        results["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False,
        )
        results["img_fields"] = ["img"]

        return results


@PIPELINES.register_module()
class LoadAnnotationFromOTEDataset:
    """
    Pipeline element that loads an annotation from a OTE Dataset on the fly.

    Expected entries in the 'results' dict that should be passed to this pipeline element are:
        results['dataset_item']: dataset_item from which to load the annotation
        results['ann_info']['label_list']: list of all labels in the project

    """

    @check_input_parameters_type()
    def __init__(self, min_size : int, with_bbox: bool = True, with_label: bool = True, with_mask: bool = False, with_seg: bool = False,
                 poly2mask: bool = True, with_text: bool = False, domain: Optional[Domain] = None):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.with_text = with_text
        self.domain = domain
        self.min_size = min_size

    @staticmethod
    def _load_bboxes(results, ann_info):
        results['bbox_fields'].append('gt_bboxes')
        results['gt_bboxes'] = copy.deepcopy(ann_info['bboxes'])
        return results

    @staticmethod
    def _load_labels(results, ann_info):
        results['gt_labels'] = copy.deepcopy(ann_info['labels'])
        return results

    @staticmethod
    def _load_masks(results, ann_info):
        results['mask_fields'].append('gt_masks')
        results['gt_masks'] = copy.deepcopy(ann_info['masks'])
        return results

    @check_input_parameters_type()
    def __call__(self, results: Dict[str, Any]):
        dataset_item = results['dataset_item']
        label_list = results['ann_info']['label_list']
        ann_info = get_annotation_mmdet_format(dataset_item, label_list, self.domain, self.min_size)
        if self.with_bbox:
            results = self._load_bboxes(results, ann_info)
            if results is None or len(results['gt_bboxes']) == 0:
                return None
        if self.with_label:
            results = self._load_labels(results, ann_info)
        if self.with_mask:
            results = self._load_masks(results, ann_info)
        return results
