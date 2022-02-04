""" This module contains the Dice performance provider. """

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


from typing import Dict, List, Optional, Tuple

from ote_sdk.entities.label import LabelEntity
from ote_sdk.entities.metrics import (
    BarChartInfo,
    BarMetricsGroup,
    ColorPalette,
    MetricsGroup,
    Performance,
    ScoreMetric,
)
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.usecases.evaluation.averaging import MetricAverageMethod
from ote_sdk.usecases.evaluation.basic_operations import (
    get_intersections_and_cardinalities,
)
from ote_sdk.usecases.evaluation.performance_provider_interface import (
    IPerformanceProvider,
)
from ote_sdk.utils.segmentation_utils import mask_from_dataset_item
from ote_sdk.utils.time_utils import timeit


class DiceAverage(IPerformanceProvider):
    """
    Computes the average Dice coefficient overall and for individual labels.

    See https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient for background information.

    To compute the Dice coefficient the shapes in the dataset items of the prediction and ground truth
    dataset are first converted to masks.

    Dice is computed by computing the intersection and union computed over the whole dataset, instead of
    computing intersection and union for individual images and then averaging.

    :param resultset: ResultSet that score will be computed for
    :param average:
        MICRO: every pixel has the same weight, regardless of label
        MACRO: compute score per label, return the average of the per-label scores
    """

    def __init__(
        self,
        resultset: ResultSetEntity,
        average: MetricAverageMethod = MetricAverageMethod.MACRO,
    ):
        self.average = average
        (
            self._overall_dice,
            self._dice_per_label,
        ) = self.__compute_dice_averaged_over_pixels(resultset, average)

    @property
    def overall_dice(self) -> ScoreMetric:
        """
        Returns the dice average as ScoreMetric
        """
        return self._overall_dice

    @property
    def dice_per_label(self) -> Dict[LabelEntity, ScoreMetric]:
        """
        Returns a dictionary mapping the label to its corresponding dice score (as ScoreMetric)
        """
        return self._dice_per_label

    def get_performance(self) -> Performance:
        score = self.overall_dice
        dashboard_metrics: Optional[List[MetricsGroup]]
        if len(self.dice_per_label) == 0:
            dashboard_metrics = None
        else:
            dashboard_metrics = [
                BarMetricsGroup(
                    metrics=list(self.dice_per_label.values()),
                    visualization_info=BarChartInfo(
                        name="Dice Average Per Label",
                        palette=ColorPalette.LABEL,
                    ),
                )
            ]
        return Performance(score=score, dashboard_metrics=dashboard_metrics)

    @classmethod
    @timeit
    def __compute_dice_averaged_over_pixels(
        cls, resultset: ResultSetEntity, average: MetricAverageMethod
    ) -> Tuple[ScoreMetric, Dict[LabelEntity, ScoreMetric]]:
        """
        computes the diced averaged over pixels
        :param average: Averaging method to use
        :param resultset: Result set to use
        :return: A tuple containing the overall DICE score, and per label DICE score
        """
        if len(resultset.prediction_dataset) == 0:
            raise ValueError("Cannot compute the DICE score of an empty result set.")

        if len(resultset.prediction_dataset) != len(resultset.ground_truth_dataset):
            raise ValueError(
                f"Prediction and ground truth dataset should have the same length. "
                f"Ground truth dataset has {len(resultset.ground_truth_dataset)} items, "
                f"prediction dataset has {len(resultset.prediction_dataset)} items"
            )
        resultset_labels = set(
            resultset.prediction_dataset.get_labels()
            + resultset.ground_truth_dataset.get_labels()
        )
        model_labels = set(
            resultset.model.configuration.label_schema.get_labels(include_empty=False)
        )
        labels = sorted(resultset_labels.intersection(model_labels))
        hard_predictions = []
        hard_references = []
        for prediction_item, reference_item in zip(
            list(resultset.prediction_dataset), list(resultset.ground_truth_dataset)
        ):
            hard_predictions.append(mask_from_dataset_item(prediction_item, labels))
            hard_references.append(mask_from_dataset_item(reference_item, labels))

        all_intersection, all_cardinality = get_intersections_and_cardinalities(
            hard_references, hard_predictions, labels
        )

        return cls.compute_dice_using_intersection_and_cardinality(
            all_intersection, all_cardinality, average
        )

    @classmethod
    def compute_dice_using_intersection_and_cardinality(
        cls,
        all_intersection: Dict[Optional[LabelEntity], int],
        all_cardinality: Dict[Optional[LabelEntity], int],
        average: MetricAverageMethod,
    ) -> Tuple[ScoreMetric, Dict[LabelEntity, ScoreMetric]]:
        """
        Computes dice score using intersection and cardinality dictionaries.
        Both dictionaries must contain the same set of keys.
        Dice score is computed by: 2 * intersection / cardinality

        :param average: Averaging method to use
        :param all_intersection: collection of intersections per label
        :param all_cardinality: collection of cardinality per label
        :return: A tuple containing the overall DICE score, and per label DICE score
        :raises KeyError: if the keys in intersection and cardinality do not match
        :raises KeyError: if the key `None` is not present in either all_intersection or all_cardinality
        :raises ValueError: if the intersection for a certain key is larger than its corresponding cardinality
        """
        dice_per_label: Dict[LabelEntity, ScoreMetric] = {}

        for label, intersection in all_intersection.items():
            cardinality = all_cardinality[label]
            dice_score = (
                cls.__compute_single_dice_score_using_intersection_and_cardinality(
                    intersection, cardinality
                )
            )

            # If label is None, then the dice score corresponds to the overall dice score
            # rather than a per-label dice score.
            # This score is calculated last because it can depend on the values in dice_per_label
            if label is not None:
                dice_per_label[label] = ScoreMetric(value=dice_score, name=label.name)

        # Set overall_dice to 0 in case the score cannot be computed
        overall_dice = ScoreMetric(value=0.0, name="Dice Average")
        if len(dice_per_label) == 0:  # dataset consists of background pixels only
            pass  # Use the default value of 0
        elif average == MetricAverageMethod.MICRO:
            overall_cardinality = all_cardinality[None]
            overall_intersection = all_intersection[None]
            dice_score = (
                cls.__compute_single_dice_score_using_intersection_and_cardinality(
                    overall_intersection, overall_cardinality
                )
            )
            overall_dice = ScoreMetric(value=dice_score, name="Dice Average")
        elif average == MetricAverageMethod.MACRO:
            scores = [item.value for item in dice_per_label.values()]
            macro_average_score = sum(scores) / len(scores)
            overall_dice = ScoreMetric(value=macro_average_score, name="Dice Average")

        return overall_dice, dice_per_label

    @staticmethod
    def __compute_single_dice_score_using_intersection_and_cardinality(
        intersection: int, cardinality: int
    ):
        """
        Computes a single dice score using intersection and cardinality.
        Dice score is computed by: 2 * intersection / cardinality
        :raises ValueError: If intersection is larger than cardinality
        """
        if intersection > cardinality:
            raise ValueError("intersection cannot be larger than cardinality")
        if cardinality == 0 and intersection == 0:
            dice_score = 0.0
        else:
            dice_score = float(2 * intersection / cardinality)
        return dice_score
