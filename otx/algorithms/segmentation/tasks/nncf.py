"""NNCF Task of OTX Segmentation."""

# Copyright (C) 2022 Intel Corporation
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

from functools import partial
from typing import Optional


from otx.algorithms.common.tasks.nncf_base import NNCFBaseTask
from otx.algorithms.segmentation.adapters.mmseg.nncf import build_nncf_model
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.optimization_parameters import OptimizationParameters
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.utils.argument_checks import check_input_parameters_type
from otx.mpa.utils.logger import get_logger

from .inference import SegmentationInferenceTask


logger = get_logger()


class SegmentationNNCFTask(NNCFBaseTask, SegmentationInferenceTask):
    @check_input_parameters_type()
    def __init__(self, task_environment: TaskEnvironment):
        super().__init__(task_environment)
        self._label_dictionary = dict(enumerate(self._labels, 1))

    def _initialize_post_hook(self, options=dict()):
        super()._initialize_post_hook(options)

        export = options.get("export", False)
        options["model_builder"] = partial(
            self.model_builder,
            nncf_model_builder=build_nncf_model,
            return_compression_ctrl=False,
            is_export=export,
        )

    def _optimize(
        self,
        dataset: DatasetEntity,
        optimization_parameters: Optional[OptimizationParameters] = None,
    ):
        results = self._run_task(
            "SegTrainer",
            mode="train",
            dataset=dataset,
            parameters=optimization_parameters,
        )
        return results

