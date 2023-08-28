"""Base Exporter for OTX tasks."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.v2.api.utils.logger import get_logger

logger = get_logger()

from typing import Any, Dict, Optional, Tuple

import torch
from mmdeploy.apis import build_task_processor
from mmdeploy.apis.core.pipeline_manager import no_mp
from mmdeploy.apis.onnx import export
from mmdeploy.backend.openvino.onnx2openvino import from_onnx
from mmdeploy.backend.openvino.utils import ModelOptimizerOptions
from mmengine.config import Config


# pylint: disable=too-many-instance-attributes
class Exporter:
    """Export class for MMEngine."""

    def __init__(
        self,
        config: Config,
        checkpoint: Optional[str],
        deploy_config: Config,
        work_dir: str,
        precision: str = "FLOAT32",
        export_type: str = "OPENVINO",
        device: str = "cpu",
    ):
        """Initialize Exporter.

        Args:
            config (Config): recipe config which contains model config
            checkpoint (str): model weights
            deploy_config (Config): deploy config which contains deploy info
            work_dir (str): path to save onnx and openvino xml file
            precision (str): whether to use mixed-precision(FP16)
            onnx_only (bool): whether to export only onnx model
            device (bool): whether to export only onnx model
        """

        self.task_processor = build_task_processor(config, deploy_config, device)
        self.checkpoint = checkpoint
        self.deploy_cfg = deploy_config

        self.model = self._get_model()
        self.input_tensor, self.input_metas = self._get_inputs()
        self.work_dir = work_dir
        self.context_info = {"deploy_cfg": deploy_config}
        if precision.upper() in ["FP16", "FLOAT16"]:
            self.deploy_cfg.backend_config.mo_options["flags"] = ["--compress_to_fp16"]
        self.onnx_only = export_type.upper() == "ONNX"

    def _get_model(self) -> torch.nn.Module:
        """Prepare torch model for exporting."""
        from otx.v2.adapters.torch.mmengine.mmdeploy.utils.utils import sync_batchnorm_2_batchnorm

        model = self.task_processor.build_pytorch_model(None)
        if self.checkpoint is not None:
            state_dict = torch.load(self.checkpoint)
            if "model" in state_dict.keys():
                state_dict = state_dict["model"]
            if "state_dict" in state_dict.keys():
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict)
        model = sync_batchnorm_2_batchnorm(model)

        # TODO: Need to investigate it why
        # NNCF compressed model lost trace context from time to time with no reason
        # even with 'torch.no_grad()'. Explicitly setting 'requires_grad' to'False'
        # makes things easier.
        for i in model.parameters():
            i.requires_grad = False
        return model

    def _get_inputs(self) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """Prepare torch model's input and input_metas."""

        input_shape = self.deploy_cfg.backend_config.model_inputs[0]["opt_shapes"]["input"]
        input_tensor = torch.randn(input_shape)
        input_metas = None
        return input_tensor, input_metas

    def export(self):
        """Export model using mmdeploy apis."""
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
