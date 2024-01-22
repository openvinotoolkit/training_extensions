# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import os
import os.path as osp
import logging as log
import importlib
from typing import Callable
from pathlib import Path

import onnx
import cv2
import torch
import numpy as np
import openvino
from omegaconf import DictConfig, OmegaConf
from mmdeploy.apis import extract_model, get_predefined_partition_cfg, torch2onnx, build_task_processor
from mmdeploy.utils import get_partition_config
from mmengine.config import Config as MMConfig

from otx.core.exporter.base import OTXModelExporter
from otx.core.utils.config import convert_conf_to_mmconfig_dict, to_tuple
from otx.core.types.export import OTXExportPrecisionType


class MMdeployExporter(OTXModelExporter):
    """_summary_

    Args:
        input_size: height, width
    """
    def __init__(
        self,
        model_builder: Callable,
        model_cfg: DictConfig,
        deploy_cfg: str | dict,
        test_pipeline: list[DictConfig],
        input_size: tuple[int, ...],
        mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
        std: tuple[float, float, float] = (1.0, 1.0, 1.0),
        resize_mode: str = "standard",
        pad_value: int = 0,
        swap_rgb: bool = False,
    ) -> None:
        super().__init__(input_size, mean, std, resize_mode, pad_value, swap_rgb)
        self._model_builder = model_builder
        model_cfg = convert_conf_to_mmconfig_dict(model_cfg, "list")
        new_pipeline = [to_tuple(OmegaConf.to_container(test_pipeline[i])) for i in range(len(test_pipeline))]
        self._model_cfg = MMConfig({"model" : model_cfg, "test_pipeline" : new_pipeline})
        self._deploy_cfg = deploy_cfg if isinstance(deploy_cfg, dict) else load_mmconfig_from_pkg(deploy_cfg)

        if input_size is not None:
            patch_input_shape(self._deploy_cfg, input_size[3], input_size[2])

    def to_openvino(
        self,
        model: torch.nn.Module,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: OTXExportPrecisionType = OTXExportPrecisionType.FP32,
        metadata: dict[tuple[str, str], str] | None = None,
    ) -> Path:
        """Export to OpenVINO Intermediate Representation format.

        Args:
            model (torch.nn.Module): pytorch model top export
            output_dir (Path): path to the directory to store export artifacts
            base_model_name (str, optional): exported model name
            precision (OTXExportPrecisionType, optional): precision of the exported model's weights
            metadata (dict[tuple[str, str], str] | None, optional): metadata to embed to the exported model.

        Returns:
            Path: path to the exported model.
        """
        onnx_path = self._cvt2onnx(model, output_dir, base_model_name)
        exported_model = openvino.convert_model(
            onnx_path,
            input=(openvino.runtime.PartialShape(self.input_size),),
        )

        metadata = {} if metadata is None else self._extend_model_metadata(metadata)
        exported_model = self._embed_openvino_ir_metadata(exported_model, metadata)
        save_path = output_dir / (base_model_name + ".xml")
        openvino.save_model(exported_model, save_path, compress_to_fp16=(precision == OTXExportPrecisionType.FP16))

        return Path(save_path)

    def to_onnx(
        self,
        model: torch.nn.Module,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: OTXExportPrecisionType = OTXExportPrecisionType.FP32,
        metadata: dict[tuple[str, str], str] | None = None,
    ) -> Path:
        """Export to ONNX format.

        Args:
            model (torch.nn.Module): pytorch model top export
            output_dir (Path): path to the directory to store export artifacts
            base_model_name (str, optional): exported model name
            precision (OTXExportPrecisionType, optional): precision of the exported model's weights
            metadata (dict[tuple[str, str],str] | None, optional): metadata to embed to the exported model.

        Returns:
            Path: path to the exported model.
        """
        save_path = self._cvt2onnx(model, output_dir, base_model_name)

        onnx_model = onnx.load(save_path)
        metadata = {} if metadata is None else self._extend_model_metadata(metadata)
        onnx_model = self._embed_onnx_metadata(onnx_model, metadata)
        if precision == OTXExportPrecisionType.FP16:
            from onnxconverter_common import float16

            onnx_model = float16.convert_float_to_float16(onnx_model)
        onnx.save(onnx_model, save_path)

        return Path(save_path)


    def _cvt2onnx(self, model: torch.nn.Module, output_dir: Path, base_model_name: str) -> str:
        model_weight_file = str(output_dir / "mmdeploy_fmt_model.pth")
        torch.save(model.state_dict(), model_weight_file)

        self._register_model_builder()
        onnx_file_name = base_model_name + ".onnx"

        log.debug(f'mmdeploy torch2onnx: \n\tmodel_cfg: {self._model_cfg}\n\tdeploy_cfg: {self._deploy_cfg}')
        torch2onnx(
            self._get_input_data(),
            output_dir,
            onnx_file_name,
            deploy_cfg=self._deploy_cfg,
            model_cfg=self._model_cfg,
            model_checkpoint=model_weight_file,
            device="cpu")

        os.remove(model_weight_file)

        # partition model
        partition_cfgs = get_partition_config(self._deploy_cfg)
        if partition_cfgs is not None:
            self.cvt_torch2onnx_partition(self._deploy_cfg, partition_cfgs, output_dir)

        return osp.join(output_dir, onnx_file_name)

    def _register_model_builder(self):
        task_processor = build_task_processor(self._model_cfg, self._deploy_cfg, "cpu")

        def helper(*args, **kwargs):
            return mmdeploy_init_model_helper(*args, **kwargs, model_builder=self._model_builder)

        task_processor.__class__.init_pytorch_model = helper

    def _get_input_data(self):
        input_data_cfg = self._deploy_cfg.get("input_data", None)

        if input_data_cfg is None:
            input_data = np.zeros((128, 128, 3), dtype=np.uint8)
        elif input_data_cfg.get("file_path") is not None:
            input_data = cv2.imread(input_data_cfg.get("file_path"))
            # image assumed to be RGB format under OTX
            input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
        else:
            input_data = np.zeros(input_data_cfg["shape"], dtype=np.uint8)

        return input_data


    def cvt_torch2onnx_partition(self, deploy_cfg, partition_cfgs, args):
        # NOTE draft version. need for exporting tilling model.
        raise NotImplementedError

        if 'partition_cfg' in partition_cfgs:
            partition_cfgs = partition_cfgs.get('partition_cfg', None)
        else:
            assert 'type' in partition_cfgs
            partition_cfgs = get_predefined_partition_cfg(
                deploy_cfg, partition_cfgs['type'])

        origin_ir_file = osp.join(args.work_dir, save_file)
        for partition_cfg in partition_cfgs:
            save_file = partition_cfg['save_file']
            save_path = osp.join(args.work_dir, save_file)
            start = partition_cfg['start']
            end = partition_cfg['end']
            dynamic_axes = partition_cfg.get('dynamic_axes', None)

            extract_model(
                origin_ir_file,
                start,
                end,
                dynamic_axes=dynamic_axes,
                save_file=save_path)


def mmdeploy_init_model_helper(*args, **kwargs):
    """Helper function for initializing a model for inference using the 'mmdeploy' library."""

    model_builder = kwargs.pop("model_builder")
    model = model_builder()

    # TODO: Need to investigate it why
    # NNCF compressed model lost trace context from time to time with no reason
    # even with 'torch.no_grad()'. Explicitly setting 'requires_grad' to'False'
    # makes things easier.
    for i in model.parameters():
        i.requires_grad = False

    return model


def patch_input_shape(deploy_cfg: MMConfig, width: int, height: int):
    """Update backend configuration with input shape information.

    This function retrieves the input size from `cfg.data.test.pipeline`,
    then sets the input shape for the backend model in `deploy_cfg`

    ```
    {
        "opt_shapes": {
            "input": [1, 3, *size]
        }
    }
    ```

    Args:
        cfg (Config): Config object containing test pipeline and other configurations.
        deploy_cfg (DeployConfig): DeployConfig object containing backend configuration.

    Returns:
        None: This function updates the input `deploy_cfg` object directly.
    """
    deploy_cfg.ir_config.input_shape = (width, height)
    deploy_cfg.backend_config.model_inputs = [
        {"opt_shapes" : dict(input=[-1, 3, height, width])}
    ]


def patch_ir_scale_factor(deploy_cfg, hyper_parameters):
    """Patch IR scale factor inplace from hyper parameters to deploy config.

    Args:
        deploy_cfg (ConfigDict): mmcv deploy config
        hyper_parameters (DetectionConfig): OTX detection hyper parameters
    """
    return
    # TODO: need to implement after tiling is implemented

    if hyper_parameters.tiling_parameters.enable_tiling:
        scale_ir_input = deploy_cfg.get("scale_ir_input", False)
        if scale_ir_input:
            tile_ir_scale_factor = hyper_parameters.tiling_parameters.tile_ir_scale_factor
            log.info(f"Apply OpenVINO IR scale factor: {tile_ir_scale_factor}")
            ir_input_shape = deploy_cfg.backend_config.model_inputs[0].opt_shapes.input
            ir_input_shape[2] = int(ir_input_shape[2] * tile_ir_scale_factor)  # height
            ir_input_shape[3] = int(ir_input_shape[3] * tile_ir_scale_factor)  # width
            deploy_cfg.ir_config.input_shape = (ir_input_shape[3], ir_input_shape[2])  # width, height
            deploy_cfg.backend_config.model_inputs = [
                ConfigDict(opt_shapes=ConfigDict(input=[1, 3, ir_input_shape[2], ir_input_shape[3]]))
            ]
            print(f"-----------------> x {tile_ir_scale_factor} = {ir_input_shape}")


def load_mmconfig_from_pkg(cfg: str) -> MMConfig:
    """Load configuration from package path as MMEngine Config format.

    Args:
        cfg (str): Package path of configuraiton.

    Returns:
        MMConfig: MMEngine Config.
    """
    config_module = importlib.import_module(cfg)
    return MMConfig.fromfile(config_module.__file__)
