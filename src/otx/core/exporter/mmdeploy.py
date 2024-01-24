# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for mmdeploy exporter used in OTX."""

from __future__ import annotations

import importlib
import logging as log
from contextlib import contextmanager
from copy import copy
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import onnx
import openvino
import torch
from mmdeploy.apis import build_task_processor, torch2onnx
from mmdeploy.utils import get_partition_config
from mmengine.config import Config as MMConfig
from omegaconf import DictConfig, OmegaConf

from mmengine.registry.default_scope import DefaultScope
from otx.core.exporter.base import OTXModelExporter
from otx.core.types.export import OTXExportPrecisionType
from otx.core.utils.config import convert_conf_to_mmconfig_dict, to_tuple


class MMdeployExporter(OTXModelExporter):
    """Exporter that uses mmdeploy and OpenVINO conversion tools.

    Args:
        model_builder (Callable): A function to build a model.
        model_cfg (DictConfig): Model config for mm framework.
        deploy_cfg (str | MMConfig): Deployment config module path or MMEngine Config object.
        test_pipeline (list[DictConfig]): A pipeline for test dataset.
        input_size (tuple[int, ...]): Input shape.
        mean (tuple[float, float, float], optional): Mean values of 3 channels. Defaults to (0.0, 0.0, 0.0).
        std (tuple[float, float, float], optional): Std values of 3 channels. Defaults to (1.0, 1.0, 1.0).
        resize_mode (Literal["crop", "standard", "fit_to_window", "fit_to_window_letterbox"], optional):
            A resize type for model preprocess. "standard" resizes iamges without keeping ratio.
            "fit_to_window" resizes images while keeping ratio.
            "fit_to_window_letterbox" resizes images and pads images to fit the size. Defaults to "standard".
        pad_value (int, optional): Padding value. Defaults to 0.
        swap_rgb (bool, optional): Whether to convert the image from BGR to RGB Defaults to False.
        max_num_detections (int, optional): Maximum number of detections per image. Defaults to 0.
    """

    def __init__(
        self,
        model_builder: Callable,
        model_cfg: DictConfig,
        deploy_cfg: str | MMConfig,
        test_pipeline: list[DictConfig],
        input_size: tuple[int, ...],
        mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
        std: tuple[float, float, float] = (1.0, 1.0, 1.0),
        resize_mode: Literal["crop", "standard", "fit_to_window", "fit_to_window_letterbox"] = "standard",
        pad_value: int = 0,
        swap_rgb: bool = False,
        max_num_detections: int = 0,
    ) -> None:
        super().__init__(input_size, mean, std, resize_mode, pad_value, swap_rgb)
        self._model_builder = model_builder
        model_cfg = convert_conf_to_mmconfig_dict(model_cfg, "list")
        new_pipeline = [to_tuple(OmegaConf.to_container(test_pipeline[i])) for i in range(len(test_pipeline))]
        self._model_cfg = MMConfig({"model": model_cfg, "test_pipeline": new_pipeline})
        self._deploy_cfg = deploy_cfg if isinstance(deploy_cfg, MMConfig) else load_mmconfig_from_pkg(deploy_cfg)

        patch_input_shape(self._deploy_cfg, input_size[3], input_size[2])
        if max_num_detections > 0:
            self._set_max_num_detections(max_num_detections)

    def _set_max_num_detections(self, max_num_detections: int) -> None:
        log.info(f"Export max_num_detections: {max_num_detections}")
        post_proc_cfg = self._deploy_cfg["codebase_config"]["post_processing"]
        post_proc_cfg["max_output_boxes_per_class"] = max_num_detections
        post_proc_cfg["keep_top_k"] = max_num_detections
        post_proc_cfg["pre_top_k"] = max_num_detections * 10

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
            str(onnx_path),
            input=(openvino.runtime.PartialShape(self.input_size),),
        )

        metadata = {} if metadata is None else self._extend_model_metadata(metadata)
        exported_model = self._embed_openvino_ir_metadata(exported_model, metadata)
        save_path = output_dir / (base_model_name + ".xml")
        openvino.save_model(exported_model, save_path, compress_to_fp16=(precision == OTXExportPrecisionType.FP16))
        onnx_path.unlink()
        log.info("Coverting to OpenVINO is done.")

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
        deploy_cfg = self._prepare_onnx_cfg()
        save_path = self._cvt2onnx(model, output_dir, base_model_name, deploy_cfg)

        onnx_model = onnx.load(str(save_path))
        metadata = {} if metadata is None else self._extend_model_metadata(metadata)
        onnx_model = self._embed_onnx_metadata(onnx_model, metadata)
        if precision == OTXExportPrecisionType.FP16:
            from onnxconverter_common import float16

            onnx_model = float16.convert_float_to_float16(onnx_model)
        onnx.save(onnx_model, str(save_path))
        log.info("Coverting to ONNX is done.")

        return save_path

    def _prepare_onnx_cfg(self) -> MMConfig:
        cfg = copy(self._deploy_cfg)
        cfg["backend_config"] = {"type": "onnxruntime"}
        return cfg

    def _cvt2onnx(
        self,
        model: torch.nn.Module,
        output_dir: Path,
        base_model_name: str,
        deploy_cfg: MMConfig | None = None,
    ) -> Path:
        onnx_file_name = base_model_name + ".onnx"
        model_weight_file = output_dir / "mmdeploy_fmt_model.pth"
        torch.save(model.state_dict(), model_weight_file)

        log.debug(f"mmdeploy torch2onnx: \n\tmodel_cfg: {self._model_cfg}\n\tdeploy_cfg: {self._deploy_cfg}")
        with use_temporary_default_scope():
            self._register_model_builder()
            torch2onnx(
                np.zeros((128, 128, 3), dtype=np.uint8),
                str(output_dir),
                onnx_file_name,
                deploy_cfg=self._deploy_cfg if deploy_cfg is None else deploy_cfg,
                model_cfg=self._model_cfg,
                model_checkpoint=str(model_weight_file),
                device="cpu",
            )

            partition_cfgs = get_partition_config(self._deploy_cfg)
            if partition_cfgs is not None:
                self.cvt_torch2onnx_partition(self._deploy_cfg, partition_cfgs)

        model_weight_file.unlink()

        return output_dir / onnx_file_name

    def _register_model_builder(self) -> None:
        task_processor = build_task_processor(self._model_cfg, self._deploy_cfg, "cpu")

        def helper(*args, **kwargs) -> torch.nn.Module:
            return mmdeploy_init_model_helper(*args, **kwargs, model_builder=self._model_builder)

        task_processor.__class__.init_pytorch_model = helper

    def cvt_torch2onnx_partition(self, deploy_cfg: MMConfig, partition_cfgs: MMConfig) -> None:
        """Partition onnx conversion."""
        raise NotImplementedError  # NOTE need for exporting tiling model.

        # if "partition_cfg" in partition_cfgs:
        #     partition_cfgs = partition_cfgs.get("partition_cfg", None)
        # else:
        #     assert "type" in partition_cfgs
        #     partition_cfgs = get_predefined_partition_cfg(deploy_cfg, partition_cfgs["type"])

        # origin_ir_file = osp.join(args.work_dir, save_file)
        # for partition_cfg in partition_cfgs:
        #     save_file = partition_cfg["save_file"]
        #     save_path = osp.join(args.work_dir, save_file)
        #     start = partition_cfg["start"]
        #     end = partition_cfg["end"]
        #     dynamic_axes = partition_cfg.get("dynamic_axes", None)

        #     extract_model(origin_ir_file, start, end, dynamic_axes=dynamic_axes, save_file=save_path)


def mmdeploy_init_model_helper(*_, **kwargs) -> torch.nn.Module:
    """Helper function for initializing a model for inference using the 'mmdeploy' library."""
    model_builder = kwargs.pop("model_builder")
    model = model_builder()

    # NOTE: Need to investigate it why
    # NNCF compressed model lost trace context from time to time with no reason
    # even with 'torch.no_grad()'. Explicitly setting 'requires_grad' to'False'
    # makes things easier.
    for i in model.parameters():
        i.requires_grad = False

    return model


def patch_input_shape(deploy_cfg: MMConfig, width: int, height: int) -> None:
    """Update backend configuration with input shape information.

    Args:
        deploy_cfg (MMConfig): Config object containing test pipeline and other configurations.
        width (int): Width of image.
        height (int): Height of image.
    """
    deploy_cfg.ir_config.input_shape = (width, height)


def patch_ir_scale_factor(deploy_cfg: MMConfig, hyper_parameters) -> None:  # noqa: ANN001, ARG001
    """Patch IR scale factor inplace from hyper parameters to deploy config.

    Args:
        deploy_cfg (ConfigDict): mmcv deploy config.
        hyper_parameters (DetectionConfig): OTX detection hyper parameters>
    """
    raise NotImplementedError  # NOTE need to implement for tiling

    # if hyper_parameters.tiling_parameters.enable_tiling:
    #     scale_ir_input = deploy_cfg.get("scale_ir_input", False)
    #     if scale_ir_input:
    #         tile_ir_scale_factor = hyper_parameters.tiling_parameters.tile_ir_scale_factor
    #         log.info(f"Apply OpenVINO IR scale factor: {tile_ir_scale_factor}")
    #         ir_input_shape = deploy_cfg.backend_config.model_inputs[0].opt_shapes.input
    #         ir_input_shape[2] = int(ir_input_shape[2] * tile_ir_scale_factor)  # height
    #         ir_input_shape[3] = int(ir_input_shape[3] * tile_ir_scale_factor)  # width
    #         deploy_cfg.ir_config.input_shape = (ir_input_shape[3], ir_input_shape[2])  # width, height
    #         print(f"-----------------> x {tile_ir_scale_factor} = {ir_input_shape}")


def load_mmconfig_from_pkg(cfg: str) -> MMConfig:
    """Load configuration from package path as MMEngine Config format.

    Args:
        cfg (str): Package path of configuraiton.

    Returns:
        MMConfig: MMEngine Config.
    """
    config_module = importlib.import_module(cfg)
    return MMConfig.fromfile(config_module.__file__)


@contextmanager
def use_temporary_default_scope() -> None:
    """Use temporary mm registry scope. After block is exited, DefaultScope is reverted as before entering block.
    
    DefaultScope is registered when executing mmdeploy. It doesn't make a problem normally but when making
    a new mm model after mmdeploy is done, it can be problematic because registry tries to find a object from default
    scope. This context manager is useful for that case.
    """
    try:
        ori_instance_dict = copy(DefaultScope._instance_dict)
        yield
    finally:
        DefaultScope._instance_dict = ori_instance_dict
