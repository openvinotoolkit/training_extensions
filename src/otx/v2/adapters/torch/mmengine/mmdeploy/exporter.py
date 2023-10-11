"""Base Exporter for OTX tasks."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from mmdeploy.apis import build_task_processor
from mmdeploy.apis.core.pipeline_manager import no_mp
from mmdeploy.apis.onnx import export
from mmdeploy.backend.openvino.onnx2openvino import from_onnx
from mmdeploy.backend.openvino.utils import ModelOptimizerOptions
from mmengine.config import Config

from otx.v2.api.utils.logger import get_logger

logger = get_logger()


# pylint: disable=too-many-instance-attributes
class Exporter:
    """Export class for MMEngine."""

    def __init__(
        self,
        config: Config,
        checkpoint: Optional[str],
        deploy_config: Config,
        work_dir: str,
        precision: Optional[str] = None,
        export_type: str = "OPENVINO",
        device: str = "cpu",
    ) -> None:
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
        if precision is None:
            precision = "FLOAT32"
        elif precision.upper() in ["FP16", "FLOAT16"]:
            self.deploy_cfg.backend_config.mo_options["flags"] = ["--compress_to_fp16"]
        self.onnx_only = export_type.upper() == "ONNX"

    def _get_model(self) -> torch.nn.Module:
        """Prepare torch model for exporting."""
        from otx.v2.adapters.torch.mmengine.mmdeploy.utils.utils import sync_batchnorm_2_batchnorm

        model = self.task_processor.build_pytorch_model(None)
        if self.checkpoint is not None:
            state_dict = torch.load(self.checkpoint)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            if "state_dict" in state_dict:
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

    def _get_inputs(self) -> Tuple[torch.Tensor, Optional[dict]]:
        """Prepare torch model's input and input_metas."""
        input_shape = self.deploy_cfg.backend_config.model_inputs[0]["opt_shapes"]["input"]
        input_tensor = torch.randn(input_shape)
        input_metas = None
        return input_tensor, input_metas

    def export(self) -> Dict[str, Dict[str, str]]:
        """Export model using mmdeploy apis."""
        results: Dict[str, Dict[str, str]] = {"outputs": {}}
        onnx_dir = Path(self.work_dir) / "onnx"
        onnx_dir.mkdir(exist_ok=True, parents=True)
        with no_mp():
            export(
                self.model,
                self.input_tensor,
                str(onnx_dir) + "/openvino",
                "onnxruntime",
                self.input_metas,
                self.context_info,
                self.deploy_cfg.ir_config.input_names,
                self.deploy_cfg.ir_config.output_names,
            )
            onnx_file = [f for f in onnx_dir.iterdir() if str(f).endswith(".onnx")][0]
            results["outputs"]["onnx"] = str(onnx_dir / onnx_file)

            if self.onnx_only:
                return results

            openvino_dir = Path(self.work_dir) / "openvino"
            openvino_dir.mkdir(exist_ok=True, parents=True)
            from_onnx(
                str(onnx_dir / onnx_file),
                str(openvino_dir),
                {self.deploy_cfg.ir_config.input_names[0]: self.input_tensor.shape},
                self.deploy_cfg.ir_config.output_names,
                ModelOptimizerOptions(self.deploy_cfg.backend_config.mo_options),
            )
            bin_file = [f for f in openvino_dir.iterdir() if str(f).endswith(".bin")][0]
            xml_file = [f for f in openvino_dir.iterdir() if str(f).endswith(".xml")][0]
            results["outputs"]["bin"] = str(openvino_dir / bin_file)
            results["outputs"]["xml"] = str(openvino_dir / xml_file)
            return results
