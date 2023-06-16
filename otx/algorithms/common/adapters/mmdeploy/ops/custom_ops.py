"""Custom patch of mmdeploy ops for openvino export."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmdeploy.core import SYMBOLIC_REWRITER
from mmdeploy.utils import get_ir_config
from torch.nn.functional import (
    GRID_SAMPLE_INTERPOLATION_MODES,
    GRID_SAMPLE_PADDING_MODES,
)
from torch.onnx import symbolic_helper
from torch.onnx._internal import jit_utils

# Remove previous registered symbolic
SYMBOLIC_REWRITER._registry._rewrite_records["squeeze"] = list()
SYMBOLIC_REWRITER._registry._rewrite_records["grid_sampler"] = list()


@SYMBOLIC_REWRITER.register_symbolic("squeeze", is_pytorch=True)
def squeeze__default(ctx, g, self, dim=None):
    """Register default symbolic function for `squeeze`.

    squeeze might be exported with IF node in ONNX, which is not supported in
    lots of backend.

    mmdeploy 0.x version do not support opset13 version squeeze, therefore this function is for
    custom patch for supporting opset13 version squeeze.

    If we adapt mmdeploy1.x version, then this function is no longer needed.
    """
    if dim is None:
        dims = []
        for i, size in enumerate(self.type().sizes()):
            if size == 1:
                dims.append(i)
    else:
        dims = [symbolic_helper._get_const(dim, "i", "dim")]

    if get_ir_config(ctx.cfg).get("opset_version", 11) >= 13:
        axes = g.op("Constant", value_t=torch.tensor(dims, dtype=torch.long))
        return g.op("Squeeze", self, axes)

    return g.op("Squeeze", self, axes_i=dims)


@SYMBOLIC_REWRITER.register_symbolic("grid_sampler", is_pytorch=True)
def grid_sampler__default(ctx, *args):
    """Register default symbolic function for `grid_sampler`.

    mmdeploy register its own ops for grid_sampler and OpenVINO doesn't support it
    Therefore this function register original GridSampler ops in pytorch which can be used from opset16
    """
    return grid_sampler(*args)


@symbolic_helper.parse_args("v", "v", "i", "i", "b")
def grid_sampler(
    g: jit_utils.GraphContext,
    input,
    grid,
    mode_enum,
    padding_mode_enum,
    align_corners,
):
    """Grid sampler ops for onnx opset16.

    This implementation is brought from torch.onnx.symbolic_opset16
    """
    mode_s = {v: k for k, v in GRID_SAMPLE_INTERPOLATION_MODES.items()}[mode_enum]
    padding_mode_s = {v: k for k, v in GRID_SAMPLE_PADDING_MODES.items()}[padding_mode_enum]
    return g.op(
        "GridSample",
        input,
        grid,
        align_corners_i=int(align_corners),
        mode_s=mode_s,
        padding_mode_s=padding_mode_s,
    )
