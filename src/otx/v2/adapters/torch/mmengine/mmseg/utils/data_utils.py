"""Collection of utils for dataset in Segmentation Task."""

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

import torch

from otx.v2.api.utils.logger import get_logger

logger = get_logger()


def get_valid_label_mask_per_batch(img_metas, num_classes):
    """Get valid label mask removing ignored classes to zero mask in a batch."""
    valid_label_mask_per_batch = []
    for _, meta in enumerate(img_metas):
        valid_label_mask = torch.Tensor([1 for _ in range(num_classes)])
        if "ignored_labels" in meta and meta["ignored_labels"]:
            valid_label_mask[meta["ignored_labels"]] = 0
        valid_label_mask_per_batch.append(valid_label_mask)
    return valid_label_mask_per_batch
