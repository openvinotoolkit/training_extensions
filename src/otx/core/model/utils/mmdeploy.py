# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import os.path as osp
import logging as log
import numpy as np
import cv2
from copy import copy
from typing import Callable
from omegaconf import DictConfig, OmegaConf
from mmdeploy.apis import extract_model, get_predefined_partition_cfg, torch2onnx, build_task_processor
from mmdeploy.utils import get_partition_config
from mmengine.config import Config as MMConfig

from otx.core.utils.config import convert_conf_to_mmconfig_dict


class MMdeployExporter:
    def __init__(
        self,
        model_builder: Callable,
        output_dir: str,
        model_cfg: DictConfig,
        deploy_cfg: DictConfig,
        test_pipeline: list[DictConfig],
        *,
        model_name: str = "model",
    ):
        self._model_builder = model_builder
        self.output_dir = output_dir
        model_cfg = convert_conf_to_mmconfig_dict(model_cfg)
        new_pipeline = [OmegaConf.to_container(test_pipeline[i]) for i in range(len(test_pipeline))]
        self._model_cfg = MMConfig({"model" : model_cfg, "test_pipeline" : new_pipeline})
        self._deploy_cfg = convert_conf_to_mmconfig_dict(deploy_cfg)
        self.model_name = model_name

    def cvt_torch2onnx(self):
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
            model_checkpoint=self._model_cfg.get("load_from", None),
            device="cpu")

        # partition model
        partition_cfgs = get_partition_config(self._deploy_cfg)

        if partition_cfgs is not None:
            self.cvt_torch2onnx_partition(self._deploy_cfg, partition_cfgs, self.output_dir)

        log.info('torch2onnx finished.')

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
        # NOTE draft version. need to modify code.
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

    def cvt_onnx2openvino(self, precision: str = "fp32"):
        deploy_cfg = copy(deploy_cfg)

        if precision == "fp16":  # NOTE use MO?
            deploy_cfg.backend_config.mo_options.flags.append("--compress_to_fp16")

        raise NotImplementedError


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
