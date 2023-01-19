# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
from collections.abc import Mapping
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter

from .utils import convert_batchnorm, is_mmdeploy_enabled, mmdeploy_init_model_helper


class NaiveExporter:
    @staticmethod
    def export2openvino(
        output_dir: str,
        model_builder: Callable,
        cfg: mmcv.Config,
        input_data: Dict[Any, Any],
        *,
        precision: str = "FP32",
        model_name: str = "model",
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        opset_version: int = 11,
        dynamic_axes: Dict[Any, Any] = {},
        mo_transforms: str = "",
    ):
        input_data = scatter(collate([input_data], samples_per_gpu=1), [-1])[0]

        model = model_builder(cfg)
        model = convert_batchnorm(model)
        model = model.cpu().eval()

        onnx_path = NaiveExporter.torch2onnx(
            output_dir,
            model,
            input_data,
            model_name=model_name,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
        )

        def get_normalize_cfg(cfg):
            def _get_normalize_cfg(cfg_):
                out = []
                if isinstance(cfg_, Mapping):
                    for key, value in cfg_.items():
                        if isinstance(value, (Mapping, list)):
                            out += _get_normalize_cfg(value)
                        if key == "type" and value == "Normalize":
                            return [cfg_]
                elif isinstance(cfg_, list):
                    for value in cfg_:
                        if isinstance(value, (Mapping, list)):
                            out += _get_normalize_cfg(value)
                return out

            cfg = _get_normalize_cfg(cfg)
            assert len(cfg) == 1
            return cfg[0]

        mo_args = {}

        normalize_cfg = get_normalize_cfg(cfg.data.test)
        if normalize_cfg.get("mean", None) is not None:
            mo_args["mean_values"] = normalize_cfg.get("mean")
        if normalize_cfg.get("std", None) is not None:
            mo_args["scale_values"] = normalize_cfg.get("std")
        if normalize_cfg.get("to_rgb", False):
            mo_args["reverse_input_channels"] = None

        if precision == "FP16":
            mo_args["compress_to_fp16"] = None
        if mo_transforms:
            mo_args["transform"] = mo_transforms

        NaiveExporter.onnx2openvino(
            output_dir,
            onnx_path,
            model_name=model_name,
            **mo_args,
        )

    @staticmethod
    def torch2onnx(
        output_dir: str,
        model: torch.nn.Module,
        input_data: Dict[Any, Any],
        *,
        model_name: str = "model",
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        opset_version: int = 11,
        dynamic_axes: Dict[Any, Any] = {},
        verbose: bool = False,
        **onnx_options,
    ) -> str:

        img_metas = input_data.get("img_metas")
        imgs = input_data.get("img")
        model.forward = partial(model.forward, img_metas=img_metas, return_loss=False)

        onnx_file_name = model_name + ".onnx"
        torch.onnx.export(
            model,
            imgs,
            os.path.join(output_dir, onnx_file_name),
            verbose=verbose,
            export_params=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            **onnx_options,
        )
        model.__dict__.pop("forward")
        return os.path.join(output_dir, onnx_file_name)

    @staticmethod
    def onnx2openvino(
        output_dir: str,
        onnx_path: str,
        *,
        model_name: str = "model",
        **openvino_options,
    ) -> Tuple[str, str]:
        from otx.mpa.utils import mo_wrapper

        mo_args = {
            "input_model": onnx_path,
            "output_dir": output_dir,
            "model_name": model_name,
        }
        mo_args.update(openvino_options)

        ret, msg = mo_wrapper.generate_ir(output_dir, output_dir, silent=False, **mo_args)
        if ret != 0:
            raise ValueError(msg)
        return (
            os.path.join(output_dir, model_name + ".xml"),
            os.path.join(output_dir, model_name + ".bin"),
        )


if is_mmdeploy_enabled():
    import mmdeploy.apis.openvino as openvino_api
    from mmdeploy.apis import build_task_processor, torch2onnx
    from mmdeploy.apis.openvino import get_input_info_from_cfg, get_mo_options_from_cfg
    from mmdeploy.core import FUNCTION_REWRITER
    from mmdeploy.utils import get_ir_config

    @FUNCTION_REWRITER.register_rewriter(
        "mmdeploy.core.optimizers.function_marker.mark_tensors",
        backend="openvino",
    )
    def remove_mark__openvino(ctx, xs: Any, *args, **kwargs):
        """Disable all marks for openvino backend

        As the Node `mark` is not able to be traced, we just return original input
        for the function `mark_tensors`.

        Args:
            xs (Any): Input structure which contains tensor.
        """
        return xs

    class MMdeployExporter:
        @staticmethod
        def export2openvino(
            output_dir: str,
            model_builder: Callable,
            cfg: mmcv.Config,
            deploy_cfg: mmcv.Config,
            *,
            model_name: str = "model",
        ):

            task_processor = build_task_processor(cfg, deploy_cfg, "cpu")

            def helper(*args, **kwargs):
                return mmdeploy_init_model_helper(*args, **kwargs, model_builder=model_builder)

            task_processor.__class__.init_pytorch_model = helper

            input_data_cfg = deploy_cfg.pop(
                "input_data",
                {"shape": (128, 128, 3), "file_path": None},
            )
            if input_data_cfg.get("file_path"):
                import cv2

                input_data = cv2.imread(input_data_cfg.get("file_path"))
                # image assumed to be RGB format under OTX
                input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
            else:
                input_data = np.zeros(input_data_cfg.shape, dtype=np.uint8)

            onnx_path = MMdeployExporter.torch2onnx(
                output_dir,
                input_data,
                cfg,
                deploy_cfg,
                model_name=model_name,
            )

            MMdeployExporter.onnx2openvino(
                output_dir,
                onnx_path,
                deploy_cfg,
                model_name=model_name,
            )

        @staticmethod
        def torch2onnx(
            output_dir: str,
            input_data: Any,
            cfg: mmcv.Config,
            deploy_cfg: mmcv.Config,
            *,
            model_name: str = "model",
        ) -> str:
            onnx_file_name = model_name + ".onnx"
            torch2onnx(
                input_data,
                output_dir,
                onnx_file_name,
                deploy_cfg=deploy_cfg,
                model_cfg=cfg,
                model_checkpoint=cfg.load_from,
                device="cpu",
            )
            return os.path.join(output_dir, onnx_file_name)

        @staticmethod
        def onnx2openvino(
            output_dir: str,
            onnx_path: str,
            deploy_cfg: Union[str, mmcv.Config],
            *,
            model_name: str = "model",
        ) -> Tuple[str, str]:
            input_info = get_input_info_from_cfg(deploy_cfg)
            output_names = get_ir_config(deploy_cfg).output_names
            mo_options = get_mo_options_from_cfg(deploy_cfg)

            if model_name:
                mo_options.args += f'--model_name "{model_name}" '

            openvino_api.from_onnx(onnx_path, output_dir, input_info, output_names, mo_options)

            return (
                os.path.join(output_dir, model_name + ".xml"),
                os.path.join(output_dir, model_name + ".bin"),
            )
