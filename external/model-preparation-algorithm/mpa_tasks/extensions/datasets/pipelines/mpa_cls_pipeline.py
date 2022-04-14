# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcls.datasets import PIPELINES
from detection_tasks.extension.utils import LoadImageFromOTEDataset


@PIPELINES.register_module()
class LoadImageFromOTEDataset(LoadImageFromOTEDataset):
    """
    Pipeline element that loads an image from a OTE Dataset on the fly. Can do conversion to float 32 if needed.

    Expected entries in the 'results' dict that should be passed to this pipeline element are:
        results['dataset_item']: dataset_item from which to load the image
        results['dataset_id']: id of the dataset to which the item belongs
        results['index']: index of the item in the dataset

    :param to_float32: optional bool, True to convert images to fp32. defaults to False
    """
