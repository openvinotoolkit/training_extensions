"""NNCF Task of OTX Segmentation."""

# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from typing import List, Optional

import otx.algorithms.segmentation.adapters.mmseg.nncf.patches  # noqa: F401  # pylint: disable=unused-import
from otx.algorithms.common.tasks.nncf_task import NNCFBaseTask
from otx.algorithms.segmentation.adapters.mmseg.nncf import build_nncf_segmentor
from otx.algorithms.segmentation.adapters.mmseg.task import MMSegmentationTask
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.metrics import (
    CurveMetric,
    InfoMetric,
    LineChartInfo,
    MetricsGroup,
    Performance,
    ScoreMetric,
    VisualizationInfo,
    VisualizationType,
)
from otx.api.entities.model import (
    ModelEntity,
)
from otx.api.entities.optimization_parameters import OptimizationParameters
from otx.api.entities.task_environment import TaskEnvironment
from otx.utils.logger import get_logger

logger = get_logger()


class SegmentationNNCFTask(NNCFBaseTask, MMSegmentationTask):  # pylint: disable=too-many-ancestors
    """SegmentationNNCFTask."""

    def __init__(self, task_environment: TaskEnvironment, output_path: Optional[str] = None):
        super().__init__()  # type: ignore [call-arg]
        super(NNCFBaseTask, self).__init__(task_environment, output_path)
        self._set_attributes_by_hyperparams()

    def configure(
        self,
        training=True,
        ir_options=None,
        export=False,
    ):
        """Configure configs for nncf task."""
        super(NNCFBaseTask, self).configure(training, ir_options, export)
        self._prepare_optimize(export)
        return self._config

    def _prepare_optimize(self, export=False):
        super()._prepare_optimize()

        self.model_builder = partial(
            self.model_builder,
            nncf_model_builder=build_nncf_segmentor,
            return_compression_ctrl=False,
            is_export=export,
        )

    def _optimize(
        self,
        dataset: DatasetEntity,
        optimization_parameters: Optional[OptimizationParameters] = None,
    ):
        results = self._train_model(dataset)

        return results

    def _optimize_post_hook(
        self,
        dataset: DatasetEntity,
        output_model: ModelEntity,
    ):
        # Get training metrics group from learning curves
        training_metrics, best_score = self._generate_training_metrics_group(self._learning_curves)
        performance = Performance(
            score=ScoreMetric(value=best_score, name=self.metric),
            dashboard_metrics=training_metrics,
        )

        logger.info(f"Final model performance: {str(performance)}")
        output_model.performance = performance

    def _generate_training_metrics_group(self, learning_curves):
        """Get Training metrics (epochs & scores).

        Parses the mmsegmentation logs to get metrics from the latest training run
        :return output List[MetricsGroup]
        """
        output: List[MetricsGroup] = []
        # Model architecture
        architecture = InfoMetric(name="Model architecture", value=self._model_name)
        visualization_info_architecture = VisualizationInfo(
            name="Model architecture", visualisation_type=VisualizationType.TEXT
        )
        output.append(
            MetricsGroup(
                metrics=[architecture],
                visualization_info=visualization_info_architecture,
            )
        )
        # Learning curves
        best_score = -1
        for key, curve in learning_curves.items():
            metric_curve = CurveMetric(xs=curve.x, ys=curve.y, name=key)
            if key == f"val/{self.metric}":
                best_score = max(curve.y)
            visualization_info = LineChartInfo(name=key, x_axis_label="Epoch", y_axis_label=key)
            output.append(MetricsGroup(metrics=[metric_curve], visualization_info=visualization_info))
        return output, best_score

    def _save_model_post_hook(self, modelinfo):
        modelinfo["input_size"] = self._input_size
