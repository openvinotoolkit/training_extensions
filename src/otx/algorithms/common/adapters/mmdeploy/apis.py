"""API of otx.algorithms.common.adapters.mmdeploy."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import time
from collections.abc import Mapping
from copy import deepcopy
from functools import partial
from subprocess import CalledProcessError  # nosec B404
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter

from .utils.mmdeploy import (
    is_mmdeploy_enabled,
    mmdeploy_init_model_helper,
    update_deploy_cfg,
)
from .utils.onnx import prepare_onnx_for_openvino
from .utils.utils import numpy_2_list

# pylint: disable=too-many-locals


class NaiveExporter:
    """NaiveExporter for non-mmdeploy export."""

    @staticmethod
    def export2backend(
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
        dynamic_axes: Optional[Dict[Any, Any]] = None,
        mo_transforms: str = "",
        export_type: str = "OPENVINO",
    ):
        """Function for exporting to openvino."""
        input_data = scatter(collate([input_data], samples_per_gpu=1), [-1])[0]

        model = model_builder(cfg)
        model = model.cpu().eval()
        dynamic_axes = dynamic_axes if dynamic_axes else dict()

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

        if "ONNX" in export_type:
            return

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
        dynamic_axes: Optional[Dict[Any, Any]] = None,
        verbose: bool = False,
        **onnx_options,
    ) -> str:
        """Function for torch to onnx exporting."""

        img_metas = input_data.get("img_metas")
        numpy_2_list(img_metas)
        imgs = input_data.get("img")
        model.forward = partial(model.forward, img_metas=img_metas, return_loss=False)

        onnx_file_name = model_name + ".onnx"
        dynamic_axes = dynamic_axes if dynamic_axes else dict()
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
        """Function for onnx to openvino exporting."""
        from otx.algorithms.common.utils import mo_wrapper

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
    from mmdeploy.apis import build_task_processor, extract_model, torch2onnx
    from mmdeploy.apis.openvino import get_input_info_from_cfg, get_mo_options_from_cfg
    from mmdeploy.core import reset_mark_function_count

    # from mmdeploy.core import FUNCTION_REWRITER
    from mmdeploy.utils import get_ir_config, get_partition_config

    class MMdeployExporter:
        """MMdeployExporter for mmdeploy exporting."""

        @staticmethod
        def export2backend(
            output_dir: str,
            model_builder: Callable,
            cfg: mmcv.Config,
            deploy_cfg: mmcv.Config,
            export_type: str,
            *,
            model_name: str = "model",
        ):
            """Function for exporting to openvino."""

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
                input_data = np.zeros(input_data_cfg["shape"], dtype=np.uint8)

            partition_cfgs = get_partition_config(deploy_cfg)
            if partition_cfgs:
                MMdeployExporter.extract_partition(
                    output_dir,
                    input_data,
                    cfg,
                    deploy_cfg,
                    export_type,
                    model_name=model_name,
                )

            onnx_paths = []
            onnx_paths.append(
                MMdeployExporter.torch2onnx(
                    output_dir,
                    input_data,
                    cfg,
                    deploy_cfg,
                    model_name=model_name,
                )
            )
            if "ONNX" not in export_type:
                for onnx_path in onnx_paths:
                    deploy_cfg_ = deepcopy(deploy_cfg)
                    update_deploy_cfg(onnx_path, deploy_cfg_)
                    MMdeployExporter.onnx2openvino(
                        output_dir,
                        onnx_path,
                        deploy_cfg_,
                    )

        @staticmethod
        def extract_partition(
            output_dir: str,
            input_data: Any,
            cfg: mmcv.Config,
            deploy_cfg: mmcv.Config,
            export_type: str,
            *,
            model_name: str = "model",
        ):
            """Function for extracting partition."""
            reset_mark_function_count()
            model_onnx = MMdeployExporter.torch2onnx(
                output_dir,
                input_data,
                cfg,
                deploy_cfg,
                model_name=model_name,
            )

            partition_cfgs = get_partition_config(deploy_cfg)
            partition_cfgs = partition_cfgs.get("partition_cfg", None)
            partition_onnx = MMdeployExporter.partition_onnx(
                output_dir,
                model_onnx,
                partition_cfgs,
            )

            if "ONNX" not in export_type:
                deploy_cfg_ = deepcopy(deploy_cfg)
                update_deploy_cfg(partition_onnx[0], deploy_cfg_)
                MMdeployExporter.onnx2openvino(
                    output_dir,
                    partition_onnx[0],
                    deploy_cfg_,
                )
                deploy_cfg["partition_config"]["apply_marks"] = False
                reset_mark_function_count()

        @staticmethod
        def torch2onnx(
            output_dir: str,
            input_data: Any,
            cfg: mmcv.Config,
            deploy_cfg: mmcv.Config,
            *,
            model_name: str = "model",
        ) -> str:
            """Function for torch to onnx exporting."""
            onnx_file_name = model_name + ".onnx"
            torch2onnx(
                input_data,
                output_dir,
                onnx_file_name,
                deploy_cfg=deploy_cfg,
                model_cfg=cfg,
                model_checkpoint=cfg.get("load_from", None),
                device="cpu",
            )
            return os.path.join(output_dir, onnx_file_name)

        @staticmethod
        def partition_onnx(
            output_dir,
            onnx_path: str,
            partition_cfgs: Union[mmcv.ConfigDict, List[mmcv.ConfigDict]],
        ) -> Tuple[str, ...]:
            """Function for parition onnx."""
            partitioned_paths = []

            if not isinstance(partition_cfgs, list):
                partition_cfgs = [partition_cfgs]

            for partition_cfg in partition_cfgs:
                save_file = partition_cfg["save_file"]
                save_path = os.path.join(output_dir, save_file)
                start = partition_cfg["start"]
                end = partition_cfg["end"]
                dynamic_axes = partition_cfg.get("dynamic_axes", None)

                extract_model(onnx_path, start, end, dynamic_axes=dynamic_axes, save_file=save_path)
                partitioned_paths.append(save_path)
            return tuple(partitioned_paths)

        @staticmethod
        def onnx2openvino(
            output_dir: str,
            onnx_path: str,
            deploy_cfg: Union[str, mmcv.Config],
            *,
            model_name: Optional[str] = None,
        ) -> Tuple[str, str]:
            """Function for onnx to openvino exporting."""

            input_info = get_input_info_from_cfg(deploy_cfg)
            output_names = get_ir_config(deploy_cfg).output_names
            mo_options = get_mo_options_from_cfg(deploy_cfg)

            if not model_name:
                model_name = os.path.basename(onnx_path).replace(".onnx", "")
            mo_options.args += f'--model_name "{model_name}" '

            onnx_ready_path = os.path.join(os.path.dirname(onnx_path), f"{model_name}_ready.onnx")
            prepare_onnx_for_openvino(onnx_path, os.path.join(os.path.dirname(onnx_path), f"{model_name}_ready.onnx"))

            try:
                openvino_api.from_onnx(onnx_ready_path, output_dir, input_info, output_names, mo_options)
            except CalledProcessError as e:
                # NOTE: mo returns non zero return code (245) even though it successfully generate IR
                cur_time = time.time()
                time_threshold = 5
                if not (
                    e.returncode == 245
                    and not {model_name + ".bin", model_name + ".xml"} - set(os.listdir(output_dir))
                    and (
                        os.path.getmtime(os.path.join(output_dir, model_name + ".bin")) - cur_time < time_threshold
                        and os.path.getmtime(os.path.join(output_dir, model_name + ".xml")) - cur_time < time_threshold
                    )
                ):
                    raise e

            return (
                os.path.join(output_dir, model_name + ".xml"),
                os.path.join(output_dir, model_name + ".bin"),
            )
