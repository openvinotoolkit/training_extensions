# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from contextlib import contextmanager
from functools import partial, partialmethod

from mmdet import core
from mmdet.core.bbox.samplers import BaseSampler
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.dense_heads.base_mask_head import BaseMaskHead
from mmdet.models.detectors.base import BaseDetector
from mmdet.models.roi_heads.base_roi_head import BaseRoIHead
from mmdet.models.roi_heads.roi_extractors import SingleRoIExtractor

from .utils import is_nncf_enabled, no_nncf_trace


#  from nncf.torch.dynamic_graph.context import get_current_context
#  print(get_current_context().is_tracing)


def nncf_wrapper(self, *args, **kwargs):
    """
    A wrapper function for nncf no tracing.
    """

    in_fn = kwargs.pop("in_fn")
    with no_nncf_trace():
        return in_fn(self, *args, **kwargs)


def wrap_mmdet_head(head_cls):
    """
    A function to wrap all the possible mmdet heads.
    """
    TARGET_CLS = (BaseDenseHead, BaseMaskHead, BaseRoIHead)
    TARGET_FNS = ("loss", "onnx_export", "get_bboxes")

    if issubclass(head_cls, TARGET_CLS):
        for func_name in TARGET_FNS:
            func = getattr(head_cls, func_name, None)
            if func is not None and "_partialmethod" not in func.__dict__:
                #  print(head_cls, func_name)
                setattr(head_cls, func_name, partialmethod(nncf_wrapper, in_fn=func))


def wrap_register_module(self, *args, **kwargs):
    """
    A function to wrap classes lazily defined such as custom ones.
    """

    in_fn = kwargs.pop("in_fn")
    module = kwargs["module"]
    wrap_mmdet_head(module)
    return in_fn(*args, **kwargs)


@contextmanager
def nncf_trace_context(self, img_metas):
    """
    A context manager for nncf graph tracing
    """

    assert (
        getattr(self, "forward_backup", None) is None
    ), "Error: one forward context inside another forward context"

    if is_nncf_enabled():
        from nncf.torch.nncf_network import NNCFNetwork

        if isinstance(self, NNCFNetwork):
            self.get_nncf_wrapped_model().forward_backup = self.forward

    def forward_nncf_trace(self, img, img_metas, **kwargs):
        return self.onnx_export(img[0], img_metas[0])

    # onnx_export in mmdet head has a bug with cuda
    self.device_backup = next(self.parameters()).device
    self.forward_backup = self.forward
    self = self.to("cpu")
    self.forward = partialmethod(forward_nncf_trace, img_metas=img_metas).__get__(self)
    yield
    self = self.to(self.device_backup)
    delattr(self, "device_backup")
    self.forward = self.forward_backup
    delattr(self, "forward_backup")

    if is_nncf_enabled() and isinstance(self, NNCFNetwork):
        self.get_nncf_wrapped_model().forward_backup = None


# add nncf context method that will be used when nncf tracing
BaseDetector.nncf_trace_context = nncf_trace_context


# wrap mmdet defined heads
for head_cls in [BaseDenseHead, BaseMaskHead, BaseRoIHead] + list(
    HEADS.module_dict.values()
):
    wrap_mmdet_head(head_cls)


# wrap register_module for mmdet's HEAD register
HEADS._register_module = partialmethod(
    wrap_register_module, in_fn=HEADS._register_module
).__get__(HEADS)


# NNCF can not trace this part with torch older 1.11.0
BaseSampler.sample = partialmethod(nncf_wrapper, in_fn=BaseSampler.sample)


core.bbox2result = partial(nncf_wrapper, in_fn=core.bbox2result)
core.bbox2roi = partial(nncf_wrapper, in_fn=core.bbox2roi)


SingleRoIExtractor.map_roi_levels = partialmethod(
    nncf_wrapper, in_fn=SingleRoIExtractor.map_roi_levels
)
