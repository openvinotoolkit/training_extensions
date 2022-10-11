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

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from otx.api.utils.segmentation_utils import mask_from_dataset_item
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.label import LabelEntity
from otx.api.utils.argument_checks import (
    check_input_parameters_type,
)

@check_input_parameters_type()
def get_annotation_mmseg_format(dataset_item: DatasetItemEntity, labels: List[LabelEntity]) -> dict:
    """
    Function to convert a OTE annotation to mmsegmentation format. This is used both
    in the OTEDataset class defined in this file as in the custom pipeline
    element 'LoadAnnotationFromOTEDataset'

    :param dataset_item: DatasetItem for which to get annotations
    :param labels: List of labels in the project
    :return dict: annotation information dict in mmseg format
    """

    gt_seg_map = mask_from_dataset_item(dataset_item, labels)
    gt_seg_map = gt_seg_map.squeeze(2).astype(np.uint8)

    ann_info = dict(gt_semantic_seg=gt_seg_map)

    return ann_info