"""NNCF Task of OTX Detection."""

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

from mpa.utils.logger import get_logger

from otx.algorithms.common.adapters.mmcv.utils import remove_from_config
from otx.algorithms.common.tasks.nncf_base import NNCFBaseTask
from otx.algorithms.detection.adapters.mmdet.nncf import build_nncf_model
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.optimization_parameters import OptimizationParameters

from .inference import DetectionInferenceTask


logger = get_logger()


class DetectionNNCFTask(NNCFBaseTask, DetectionInferenceTask):
    def _initialize_post_hook(self, options=dict()):
        super()._initialize_post_hook(options)

        export = options.get("export", False)
        options["model_builder"] = partial(
            self.model_builder,
            nncf_model_builder=build_nncf_model,
            return_compression_ctrl=False,
            is_export=export,
        )

        # do not configure regularization
        if (
            "l2sp_weight" in self._recipe_cfg.model
            or "l2sp_weight" in self._model_cfg.model
        ):
            remove_from_config(self._recipe_cfg.model, "l2sp_weight")
            remove_from_config(self._model_cfg.model, "l2sp_weight")

    def _optimize(
        self,
        dataset: DatasetEntity,
        optimization_parameters: Optional[OptimizationParameters] = None,
    ):
        results = self._run_task(
            "DetectionTrainer",
            mode="train",
            dataset=dataset,
            parameters=optimization_parameters,
        )
        return results

    def _update_modelinfo_to_save(self, modelinfo):
        config = modelinfo["meta"]["config"]
        if hasattr(config.model, "bbox_head") and hasattr(
            config.model.bbox_head, "anchor_generator"
        ):
            if getattr(
                config.model.bbox_head.anchor_generator,
                "reclustering_anchors",
                False,
            ):
                generator = config.model.bbox_head.anchor_generator
                modelinfo["anchors"] = {
                    "heights": generator.heights,
                    "widths": generator.widths,
                }

        modelinfo["confidence_threshold"] = self.confidence_threshold
