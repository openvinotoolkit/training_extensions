# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for Model API segmentation tiler."""
from __future__ import annotations

import numpy as np
from model_api.models.utils import ImageResultWithSoftPrediction
from model_api.tilers import Tiler


class SegTiler(Tiler):
    """Tiler for segmentation models."""

    def _postprocess_tile(
        self,
        predictions: ImageResultWithSoftPrediction,
        coord: list[int],
    ) -> dict:
        """Postprocess the tile predictions.

        Args:
            predictions (ImageResultWithSoftPrediction): predictions from SegmentationModel
            coord (list[int]): coordinates of the tile

        Returns:
            dict: postprocessed predictions
        """
        output_dict = {}
        output_dict["coord"] = coord
        output_dict["masks"] = predictions.soft_prediction
        return output_dict

    def _merge_results(self, results: list[dict], shape: tuple[int, int, int]) -> ImageResultWithSoftPrediction:
        """Merge the results from all tiles.

        Args:
            results (list[dict]): list of tile predictions
            shape (tuple[int, int, int]): shape of the original image

        Returns:
            ImageResultWithSoftPrediction: merged predictions
        """
        height, width = shape[:2]
        num_classes = len(self.model.labels)
        full_logits_mask = np.zeros((height, width, num_classes), dtype=np.float32)
        vote_mask = np.zeros((height, width), dtype=np.int32)
        for result in results:
            x1, y1, x2, y2 = result["coord"]
            mask = result["masks"]
            vote_mask[y1:y2, x1:x2] += 1
            full_logits_mask[y1:y2, x1:x2, :] += mask[: y2 - y1, : x2 - x1, :]

        full_logits_mask = full_logits_mask / vote_mask[:, :, None]
        index_mask = full_logits_mask.argmax(2)
        return ImageResultWithSoftPrediction(
            resultImage=index_mask,
            soft_prediction=full_logits_mask,
            feature_vector=np.array([]),
            saliency_map=np.array([]),
        )
