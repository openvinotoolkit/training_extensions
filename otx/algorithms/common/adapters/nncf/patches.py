# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from contextlib import contextmanager
from functools import partial

import torch
from nncf.torch.dynamic_graph.io_handling import replicate_same_tensors

from .patchers import NO_TRACE_PATCHER, TRACE_PATCHER


NO_TRACE_PATCHER.patch("mpa.modules.utils.export_helpers.get_saliency_map")
NO_TRACE_PATCHER.patch("mpa.modules.utils.export_helpers.get_feature_vector")


@contextmanager
def nncf_trace_context(self, img_metas, nncf_compress_postprocessing=True):
    """
    A context manager for nncf graph tracing
    """

    # onnx_export in mmdet head has a bug on GPU
    # it must be on CPU
    device_backup = next(self.parameters()).device
    self = self.to("cpu")
    # HACK
    # temporarily change current context as onnx export context
    # to trace network
    onnx_backup = torch.onnx.utils.__IN_ONNX_EXPORT
    torch.onnx.utils.__IN_ONNX_EXPORT = True
    # backup forward
    forward_backup = self.forward
    if nncf_compress_postprocessing:
        self.forward = partial(self.forward, img_metas=img_metas)
        #  self.forward = partial(self.forward, img_metas=img_metas, return_loss=False)
    else:
        self.forward = partial(self.forward_dummy)

    yield

    # make everything normal
    self.forward = forward_backup
    torch.onnx.utils.__IN_ONNX_EXPORT = onnx_backup
    self = self.to(device_backup)


def nncf_train_step(self, data, optimizer):
    with self._compressed_context as ctx:
        ctx.base_module_thread_local_replica = self
        _, data = replicate_same_tensors(((), data))
        if not self._in_user_dummy_forward:
            # If a user supplies own dummy forward, he is responsible for
            # correctly wrapping inputs inside it as well.
            _, data = self._strip_traced_tensors((), data)
            _, data = self._wrap_inputs_fn((), data)
        retval = self.get_nncf_wrapped_model().train_step(data, optimizer)
        retval = replicate_same_tensors(retval)
        if not self._in_user_dummy_forward:
            retval = self._wrap_outputs_fn(retval)

    # TODO: deal with kd_loss_handler in forward method of  NNCFNEtwork
    return retval
