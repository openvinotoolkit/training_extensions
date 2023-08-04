"""Custom patch of mmdeploy ops for openvino export."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmdeploy.core import SYMBOLIC_REWRITER
from mmdeploy.utils import get_ir_config
from torch.onnx import symbolic_helper

# Remove previous registered symbolic
SYMBOLIC_REWRITER._registry._rewrite_records["squeeze"] = list()


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
