# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import os
import os.path as osp
import logging as log
import time
from copy import copy
from typing import Callable
from subprocess import CalledProcessError
from pathlib import Path

import cv2
import numpy as np
import mmdeploy.apis.openvino as openvino_api
from omegaconf import DictConfig, OmegaConf
from mmdeploy.apis import extract_model, get_predefined_partition_cfg, torch2onnx, build_task_processor
from mmdeploy.apis.openvino import get_input_info_from_cfg, get_mo_options_from_cfg
from mmdeploy.utils import get_ir_config, get_partition_config
from mmengine.config import Config as MMConfig

from otx.core.utils.config import convert_conf_to_mmconfig_dict, to_tuple
from otx.core.types.export import OTXExportPrecisionType
from .onnx import prepare_onnx_for_openvino


class MMdeployExporter:
    """_summary_

    Args:
        input_size: height, width
    """
    def __init__(
        self,
        model_builder: Callable,
        output_dir: Path,
        model_cfg: DictConfig,
        deploy_cfg: dict,
        test_pipeline: list[DictConfig],
        input_size: tuple[int, int] | None = None,
        *,
        model_name: str = "model",
    ):
        self._model_builder = model_builder
        self.output_dir = output_dir
        model_cfg = convert_conf_to_mmconfig_dict(model_cfg, "list")
        new_pipeline = [to_tuple(OmegaConf.to_container(test_pipeline[i])) for i in range(len(test_pipeline))]
        self._model_cfg = MMConfig({"model" : model_cfg, "test_pipeline" : new_pipeline})
        self._deploy_cfg = deploy_cfg
        self.model_name = model_name

        patch_input_preprocessing(model_cfg, self._deploy_cfg)
        if input_size is not None:
            patch_input_shape(self._deploy_cfg, input_size[3], input_size[2])

    def cvt_torch2onnx(self) -> str:
        log.info(f'torch2onnx: \n\tmodel_cfg: {self._model_cfg}\n\tdeploy_cfg: {self._deploy_cfg}')
        input_data = self._get_input_data()
        
        self._register_model_builder()

        onnx_file_name = self.model_name + ".onnx"
        torch2onnx(
            input_data,
            self.output_dir,
            onnx_file_name,
            deploy_cfg=self._deploy_cfg,
            model_cfg=self._model_cfg,
            model_checkpoint=self._model_cfg.model.pop("load_from", None),
            device="cpu")

        # partition model
        partition_cfgs = get_partition_config(self._deploy_cfg)

        if partition_cfgs is not None:
            self.cvt_torch2onnx_partition(self._deploy_cfg, partition_cfgs, self.output_dir)

        log.info('torch2onnx finished.')

        return osp.join(self.output_dir, onnx_file_name)

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

    def cvt_onnx2openvino(self, onnx_path: str, precision: OTXExportPrecisionType = OTXExportPrecisionType.FP32):
        deploy_cfg = copy(self._deploy_cfg)

        if precision == OTXExportPrecisionType.FP16:
            deploy_cfg.backend_config.mo_options.flags.append("--compress_to_fp16")

        input_info = get_input_info_from_cfg(deploy_cfg)
        output_names = get_ir_config(deploy_cfg).output_names
        mo_options = get_mo_options_from_cfg(deploy_cfg)

        mo_options.args += f'--model_name "{self.model_name}" '

        onnx_ready_path = osp.join(osp.dirname(onnx_path), f"{self.model_name}_ready.onnx")
        prepare_onnx_for_openvino(onnx_path, osp.join(osp.dirname(onnx_path), f"{self.model_name}_ready.onnx"))

        try:
            openvino_api.from_onnx(onnx_ready_path, self.output_dir, input_info, output_names, mo_options)
        except CalledProcessError as e:
            # NOTE: mo returns non zero return code (245) even though it successfully generate IR
            cur_time = time.time()
            time_threshold = 5
            if not (
                e.returncode == 245
                and not {self.model_name + ".bin", self.model_name + ".xml"} - set(os.listdir(self.output_dir))
                and (
                    osp.getmtime(osp.join(self.output_dir, self.model_name + ".bin")) - cur_time < time_threshold
                    and osp.getmtime(osp.join(self.output_dir, self.model_name + ".xml")) - cur_time < time_threshold
                )
            ):
                raise e

        return (
            osp.join(self.output_dir, self.model_name + ".xml"),
            osp.join(self.output_dir, self.model_name + ".bin"),
        )



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


def patch_input_preprocessing(model_cfg: MMConfig, deploy_cfg: MMConfig):
    """Update backend configuration with input preprocessing options.

    - If `"to_rgb"` in Normalize config is truthy, it adds `"--reverse_input_channels"` as a flag.

    The function then sets default values for the backend configuration in `deploy_cfg`.

    Args:
        cfg (mmcv.ConfigDict): Config object containing test pipeline and other configurations.
        deploy_cfg (mmcv.ConfigDict): DeployConfig object containing backend configuration.

    Returns:
        None: This function updates the input `deploy_cfg` object directly.
    """
    normalize_cfg = model_cfg.data_preprocessor

    # Set options based on Normalize config
    options = {
        "flags": ["--reverse_input_channels"] if normalize_cfg.get("to_rgb", False) else [],
        "args": {
            "--mean_values": list(normalize_cfg.get("mean", [])),
            "--scale_values": list(normalize_cfg.get("std", [])),
        },
    }

    # Set default backend configuration
    mo_options = deploy_cfg.backend_config.get("mo_options", MMConfig())
    mo_options = MMConfig() if mo_options is None else mo_options
    mo_options.args = mo_options.get("args", MMConfig())
    mo_options.flags = mo_options.get("flags", [])

    # Override backend configuration with options from Normalize config
    mo_options.args.update(options["args"])
    mo_options.flags = list(set(mo_options.flags + options["flags"]))

    deploy_cfg.backend_config.mo_options = mo_options


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
        MMConfig(dict(opt_shapes=MMConfig(dict(input=[-1, 3, height, width]))))
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
