# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Customised MAP metric for instance segmentation."""

from __future__ import annotations

from typing import Any

import pycocotools.mask as mask_utils
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class OTXInstSegMeanAveragePrecision(MeanAveragePrecision):
    """Mean Average Precision for Instance Segmentation.

    This metric computes RLE directly from torch.Tensor masks to
    accelerate the computation.
    """

    def encode_rle(self, mask: torch.Tensor) -> dict:
        """Encodes a mask into RLE format.

        Rewrite of https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py

        Example:
            Given M=[0 0 1 1 1 0 1] the RLE counts is [2 3 1 1].
            Or for M=[1 1 1 1 1 1 0] the RLE counts is [0 6 1].

        Args:
            mask (torch.Tensor): A binary mask (0 or 1) of shape (H, W).

        Returns:
            dict: A dictionary with keys "counts" and "size".
        """
        rle = {"counts": [], "size": list(mask.shape)}
        device = mask.device
        vector = mask.t().ravel()
        diffs = torch.diff(vector)
        next_diffs = torch.where(diffs != 0)[0] + 1

        counts = torch.diff(
            torch.cat(
                (
                    torch.tensor([0], device=device),
                    next_diffs,
                    torch.tensor([len(vector)], device=device),
                ),
            ),
        )

        # odd counts are always the numbers of zeros
        if vector[0] == 1:
            counts = torch.cat((torch.tensor([0], device=device), counts))

        rle["counts"] = counts.tolist()
        return rle

    def _get_safe_item_values(
        self,
        item: dict[str, Any],
        warn: bool = False,
    ) -> tuple:
        """Convert masks to RLE format.

        Args:
            item: input dictionary containing the boxes or masks
            warn: whether to warn if the number of boxes or masks exceeds the max_detection_thresholds

        Returns:
            RLE encoded masks
        """
        if "segm" in self.iou_type:
            masks = []
            for mask in item["masks"]:
                rle = self.encode_rle(mask)
                rle = mask_utils.frPyObjects(rle, *rle["size"])
                masks.append((tuple(rle["size"]), rle["counts"]))
        return None, tuple(masks)
