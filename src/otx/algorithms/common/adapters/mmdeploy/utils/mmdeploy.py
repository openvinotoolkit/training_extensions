"""Functions for mmdeploy adapters."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import importlib

import onnx


def is_mmdeploy_enabled():
    """Checks if the 'mmdeploy' Python module is installed and available for use.

    Returns:
        bool: True if 'mmdeploy' is installed, False otherwise.

    Example:
        >>> is_mmdeploy_enabled()
        True
    """
    return importlib.util.find_spec("mmdeploy") is not None


def mmdeploy_init_model_helper(ctx, model_checkpoint=None, cfg_options=None, **kwargs):
    """Helper function for initializing a model for inference using the 'mmdeploy' library."""

    model_builder = kwargs.pop("model_builder")
    model = model_builder(
        ctx.model_cfg,
        checkpoint=model_checkpoint,
        device=ctx.device,
        cfg_options=cfg_options,
    )

    # TODO: Need to investigate it why
    # NNCF compressed model lost trace context from time to time with no reason
    # even with 'torch.no_grad()'. Explicitly setting 'requires_grad' to'False'
    # makes things easier.
    for i in model.parameters():
        i.requires_grad = False

    return model


def update_deploy_cfg(onnx_path, deploy_cfg, mo_options=None):
    """Update the 'deploy_cfg' configuration file based on the ONNX model specified by 'onnx_path'."""

    from mmdeploy.utils import get_backend_config, get_ir_config

    onnx_model = onnx.load(onnx_path)
    ir_config = get_ir_config(deploy_cfg)
    get_backend_config(deploy_cfg)

    # update input
    input_names = [i.name for i in onnx_model.graph.input]
    ir_config["input_names"] = input_names

    # update output
    output_names = [i.name for i in onnx_model.graph.output]
    ir_config["output_names"] = output_names

    # update mo options
    mo_options = mo_options if mo_options else dict()
    deploy_cfg.merge_from_dict({"backend_config": {"mo_options": mo_options}})


if is_mmdeploy_enabled():
    # fmt: off
    # FIXME: openvino pot library adds stream handlers to root logger
    # which makes annoying duplicated logging
    from mmdeploy.utils import get_root_logger
    get_root_logger().propagate = False
    # fmt: on
