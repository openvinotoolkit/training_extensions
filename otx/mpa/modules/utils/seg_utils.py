# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch


def get_valid_label_mask_per_batch(img_metas, num_classes):
    valid_label_mask_per_batch = []
    for _, meta in enumerate(img_metas):
        valid_label_mask = torch.Tensor([1 for _ in range(num_classes)])
        if "ignored_labels" in meta and meta["ignored_labels"]:
            valid_label_mask[meta["ignored_labels"]] = 0
        valid_label_mask_per_batch.append(valid_label_mask)
    return valid_label_mask_per_batch
