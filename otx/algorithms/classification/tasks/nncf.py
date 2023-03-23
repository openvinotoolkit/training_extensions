"""NNCF Task for OTX Classification."""

# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from functools import partial
from typing import List, Optional

import otx.algorithms.classification.adapters.mmcls.nncf.patches  # noqa: F401  # pylint: disable=unused-import
import otx.algorithms.classification.adapters.mmcls.nncf.registers  # noqa: F401  # pylint: disable=unused-import
from otx.algorithms.classification.adapters.mmcls.nncf.builder import (
    build_nncf_classifier,
)
from otx.algorithms.common.tasks.nncf_base import NNCFBaseTask
from otx.algorithms.common.utils.logger import get_logger
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.metrics import (
    CurveMetric,
    LineChartInfo,
    LineMetricsGroup,
    MetricsGroup,
    Performance,
    ScoreMetric,
)
from otx.api.entities.model import ModelEntity  # ModelStatus
from otx.api.entities.optimization_parameters import OptimizationParameters

from .inference import ClassificationInferenceTask

logger = get_logger()


class ClassificationNNCFTask(NNCFBaseTask, ClassificationInferenceTask):  # pylint: disable=too-many-ancestors
    """ClassificationNNCFTask."""

    def _initialize_post_hook(self, options=None):
        super()._initialize_post_hook(options)

        export = options.get("export", False)
        options["model_builder"] = partial(
            self.model_builder,
            nncf_model_builder=build_nncf_classifier,
            return_compression_ctrl=False,
            is_export=export,
        )

    def _optimize(
        self,
        dataset: DatasetEntity,
        optimization_parameters: Optional[OptimizationParameters] = None,
    ):
        results = self._run_task(
            "ClsTrainer",
            mode="train",
            dataset=dataset,
            parameters=optimization_parameters,
        )
        return results

    def _optimize_post_hook(
        self,
        dataset: DatasetEntity,
        output_model: ModelEntity,
    ):
        # Get training metrics group from learning curves
        training_metrics, final_acc = self._generate_training_metrics_group(self._learning_curves)
        performance = Performance(
            score=ScoreMetric(value=final_acc, name="accuracy"),
            dashboard_metrics=training_metrics,
        )

        logger.info(f"Final model performance: {str(performance)}")
        output_model.performance = performance

    def _generate_training_metrics_group(self, learning_curves):
        """Parses the classification logs to get metrics from the latest training run.

        :return output List[MetricsGroup]
        """
        output: List[MetricsGroup] = []

        if self._multilabel:
            metric_key = "val/accuracy-mlc"
        elif self._hierarchical:
            metric_key = "val/MHAcc"
        else:
            metric_key = "val/accuracy_top-1"

        # Learning curves
        best_acc = -1
        if learning_curves is None:
            return output

        for key, curve in learning_curves.items():
            metric_curve = CurveMetric(xs=curve.x, ys=curve.y, name=key)
            if key == metric_key:
                best_acc = max(curve.y)
            visualization_info = LineChartInfo(name=key, x_axis_label="Timestamp", y_axis_label=key)
            output.append(LineMetricsGroup(metrics=[metric_curve], visualization_info=visualization_info))

        return output, best_acc
