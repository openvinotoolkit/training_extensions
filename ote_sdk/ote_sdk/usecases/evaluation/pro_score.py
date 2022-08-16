""" This module contains the Per-Region Overlap (PRO) performance provider. """

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Tuple

import cv2
import numpy as np
from sklearn.metrics import recall_score

from ote_sdk.entities.metrics import Performance, ScoreMetric
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.usecases.evaluation.performance_provider_interface import (
    IPerformanceProvider,
)
from ote_sdk.utils.segmentation_utils import mask_from_dataset_item


class PROScore(IPerformanceProvider):
    """Computes the Per-Region Overlap Score.

    Args:
        resultset (ResultSetEntity): The result set over which the PRO score should be computed.
    """

    def __init__(self, resultset: ResultSetEntity):
        self.pro = self._compute_pro_averaged_over_regions(resultset)

    @classmethod
    def _compute_pro_averaged_over_regions(
        cls, resultset: ResultSetEntity
    ) -> ScoreMetric:
        """Compute the PRO metrics averaged over all pixel regions in the dataset.

        Args:
            resultset (ResultSetEntity): The result set over which the PRO score should be computed.
        Returns:
            ScoreMetric: Metric object containing the computed PRO score.
        """
        # Collect labels
        resultset_labels = set(
            resultset.prediction_dataset.get_labels()
            + resultset.ground_truth_dataset.get_labels()
        )
        model_labels = set(
            resultset.model.configuration.get_label_schema().get_labels(
                include_empty=False
            )
        )
        labels = sorted(resultset_labels.intersection(model_labels))
        total_pro = 0.0
        total_regions = 0
        # Get masks and compute pro score for each image
        for pred_item, gt_item in zip(
            list(resultset.prediction_dataset), list(resultset.ground_truth_dataset)
        ):
            pred_mask = mask_from_dataset_item(pred_item, labels).squeeze()
            gt_mask = mask_from_dataset_item(gt_item, labels)

            pro, n_regions = cls.compute_pro(pred_mask, gt_mask)
            total_pro += pro * n_regions
            total_regions += n_regions

        # average pro score across all regions in all images
        pro_score = total_pro / total_regions

        return ScoreMetric(value=pro_score, name="PRO Metric")

    @staticmethod
    def compute_pro(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[float, int]:
        """Compute the PRO score for a single image.

        Args:
            pred_mask (np.ndarray): Mask prediction for a single image.
            gt_mask (np.ndarray): Ground truth mask for a single image.
        Returns:
            float: PRO score for the current image.
            int: Number of regions in the image with the positive label (needed for averaging across regions).
        """
        _, gt_comps = cv2.connectedComponents(gt_mask)
        n_comps = len(np.unique(gt_comps))

        # When the image contains background only, the PRO score is always 1.0
        if n_comps == 1:
            return 1.0, 0

        # assign component labels to predicted mask
        labeled_predictions = gt_comps.copy()
        labeled_predictions[np.where(pred_mask == 0)] = 0

        # The PRO score is equal to the average recall of the components in the ground truth mask
        pro = recall_score(
            gt_comps.flatten(),
            labeled_predictions.flatten(),
            labels=np.arange(1, n_comps),
            average="macro",
        )
        return pro, n_comps - 1

    def get_performance(self) -> Performance:
        """Return the Performance object."""
        score = self.pro
        return Performance(score=score, dashboard_metrics=None)
