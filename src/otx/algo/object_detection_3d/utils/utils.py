# ------------------------------------------------------------------------
# DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""Utilities for bounding box manipulation and GIoU.
"""
import torch
from torch import Tensor


class NestedTensor:
    def __init__(self, tensors, mask: Tensor):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def box_cxcylrtb_to_xyxy(x):
    x_c, y_c, l, r, t, b = x.unbind(-1)
    bb = [(x_c - l), (y_c - t), (x_c + r), (y_c + b)]
    return torch.stack(bb, dim=-1)
