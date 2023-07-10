"""Utils for Action recognition OpenVINO export task."""

# Copyright (C) 2023 Intel Corporation
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

from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from mmcv.runner import BaseModule
from mmcv.utils import Config
from mmdeploy.apis import build_task_processor
from mmdeploy.apis.core.pipeline_manager import no_mp
from mmdeploy.apis.onnx import export
from mmdeploy.backend.openvino.onnx2openvino import from_onnx
from mmdeploy.backend.openvino.utils import ModelOptimizerOptions

from otx.algorithms.action.adapters.mmaction.models import AVAFastRCNN


def _convert_sync_batch_to_normal_batch(module: BaseModule):
    """Convert the syncBNs into normal BN3ds."""
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm3d(
            module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats
        )
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_sync_batch_to_normal_batch(child))
    del module
    return module_output


# pylint: disable=too-many-instance-attributes
class Exporter:
    """Export class for action recognition model using mmdeploy framework."""

    def __init__(
        self,
        recipe_cfg: Config,
        weights: OrderedDict,
        deploy_cfg: Config,
        work_dir: str,
        half_precision: bool,
        onnx_only: bool,
    ):
        """Initialize Exporter.

        Args:
            recipe_cfg (Config): recipe config which contains model config
            weights (str): model weights
            deploy_cfg (Config): deploy config which contains deploy info
            work_dir (str): path to save onnx and openvino xml file
            half_precision (bool): whether to use half-precision(FP16)
            onnx_only (bool): whether to export only onnx model
        """

        self.task_processor = build_task_processor(recipe_cfg, deploy_cfg, "cpu")
        self.weights = weights

        self.deploy_cfg = deploy_cfg

        self.model = self._get_model()
        self.input_tensor, self.input_metas = self._get_inputs()
        self.work_dir = work_dir
        self.context_info = {"deploy_cfg": deploy_cfg}
        if half_precision:
            self.deploy_cfg.backend_config.mo_options["flags"] = ["--compress_to_fp16"]
        self.onnx_only = onnx_only

    def _get_model(self) -> torch.nn.Module:
        """Prepare torch model for exporting."""

        model = self.task_processor.init_pytorch_model(None)
        model.load_state_dict(self.weights)
        if isinstance(model, AVAFastRCNN):
            model.patch_for_export()
        return _convert_sync_batch_to_normal_batch(model)

    def _get_inputs(self) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """Prepare torch model's input and input_metas."""

        height, width = self.deploy_cfg.backend_config.model_inputs[0]["opt_shapes"]["input"][-2:]
        if isinstance(self.model, AVAFastRCNN):
            input_tensor = torch.randn(1, 3, 32, height, width)
            input_metas = {
                "img_metas": [
                    [
                        {
                            "ori_shape": (height, width),
                            "img_shape": (height, width),
                            "pad_shape": (height, width),
                            "scale_factor": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                        }
                    ]
                ]
            }
        else:
            input_tensor = torch.randn(1, 1, 3, 32, height, width)
            input_metas = None
        return input_tensor, input_metas

    def export(self):
        """Export action model using mmdeploy apis."""

        with no_mp():
            export(
                self.model,
                self.input_tensor,
                self.work_dir,
                "onnxruntime",
                self.input_metas,
                self.context_info,
                self.deploy_cfg.ir_config.input_names,
                self.deploy_cfg.ir_config.output_names,
            )

            if self.onnx_only:
                return

            from_onnx(
                self.work_dir + ".onnx",
                self.work_dir.replace("openvino", ""),
                {self.deploy_cfg.ir_config.input_names[0]: self.input_tensor.shape},
                self.deploy_cfg.ir_config.output_names,
                ModelOptimizerOptions(self.deploy_cfg.backend_config.mo_options),
            )
