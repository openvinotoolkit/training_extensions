"""Patch mmdet."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=protected-access,redefined-outer-name

from functools import partial

from mmdet.core.bbox.assigners.base_assigner import BaseAssigner
from mmdet.core.bbox.builder import BBOX_ASSIGNERS, BBOX_SAMPLERS
from mmdet.core.bbox.samplers import BaseSampler
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.dense_heads.base_mask_head import BaseMaskHead
from mmdet.models.detectors.base import BaseDetector
from mmdet.models.roi_heads.base_roi_head import BaseRoIHead
from mmdet.models.roi_heads.bbox_heads.bbox_head import BBoxHead
from mmdet.models.roi_heads.bbox_heads.sabl_head import SABLHead
from mmdet.models.roi_heads.mask_heads.fcn_mask_head import FCNMaskHead

from otx.algorithms.common.adapters.mmdeploy.utils import is_mmdeploy_enabled
from otx.algorithms.common.adapters.nncf import (
    NNCF_PATCHER,
    is_in_nncf_tracing,
    nncf_trace_wrapper,
    no_nncf_trace_wrapper,
)
from otx.algorithms.common.adapters.nncf.patches import nncf_trace_context

HEADS_TARGETS = dict(
    classes=(
        BaseDenseHead,
        BaseMaskHead,
        BaseRoIHead,
        FCNMaskHead,
        BBoxHead,
        SABLHead,
    ),
    fn_names=("loss", "onnx_export", "get_bboxes", "get_seg_masks"),
)

BBOX_ASSIGNERS_TARGETS = dict(
    classes=(BaseAssigner,),
    fn_names=("assign",),
)

SAMPLERS_TARGETS = dict(
    classes=(BaseSampler,),
    fn_names=("sample",),
)


def _should_wrap(obj_cls, fn_name, targets):
    classes = targets["classes"]
    fn_names = targets["fn_names"]

    if obj_cls is None:
        return False
    if fn_name not in fn_names:
        return False
    if issubclass(obj_cls, classes) and getattr(obj_cls, fn_name, None):
        return True
    return False


def _wrap_mmdet_head(obj_cls):
    for fn_name in HEADS_TARGETS["fn_names"]:
        if _should_wrap(obj_cls, fn_name, HEADS_TARGETS):
            NNCF_PATCHER.patch((obj_cls, fn_name), no_nncf_trace_wrapper)
            # 'onnx_export' method calls 'forward' method which need to be traced
            NNCF_PATCHER.patch((obj_cls, "forward"), nncf_trace_wrapper)


def _wrap_mmdet_bbox_assigner(obj_cls):
    for fn_name in BBOX_ASSIGNERS_TARGETS["fn_names"]:
        if _should_wrap(obj_cls, fn_name, BBOX_ASSIGNERS_TARGETS):
            NNCF_PATCHER.patch((obj_cls, fn_name), no_nncf_trace_wrapper)


def _wrap_mmdet_sampler(obj_cls):
    for fn_name in SAMPLERS_TARGETS["fn_names"]:
        if _should_wrap(obj_cls, fn_name, SAMPLERS_TARGETS):
            NNCF_PATCHER.patch((obj_cls, fn_name), no_nncf_trace_wrapper)


# pylint: disable=invalid-name,unused-argument
def _wrap_register_module(self, fn, *args, **kwargs):
    """A function to wrap classes lazily defined such as custom ones."""

    module = kwargs.get("module", args[0] if args else None)
    assert module is not None
    _wrap_mmdet_head(module)
    _wrap_mmdet_bbox_assigner(module)
    _wrap_mmdet_sampler(module)
    return fn(*args, **kwargs)


# for mmdet defined heads
for head_cls in [BaseDenseHead, BaseMaskHead, BaseRoIHead] + list(HEADS.module_dict.values()):
    _wrap_mmdet_head(head_cls)

# for mmdet defined bbox assigners
for bbox_assigner_cls in [BaseAssigner] + list(BBOX_ASSIGNERS.module_dict.values()):
    _wrap_mmdet_bbox_assigner(bbox_assigner_cls)

# for mmdet defined samplers
# NNCF can not trace this part with torch older 1.11.0
for sampler_cls in [BaseSampler] + list(BBOX_SAMPLERS.module_dict.values()):
    _wrap_mmdet_sampler(sampler_cls)

# for custom defined
NNCF_PATCHER.patch(HEADS._register_module, _wrap_register_module)
NNCF_PATCHER.patch(BBOX_ASSIGNERS._register_module, _wrap_register_module)
NNCF_PATCHER.patch(BBOX_SAMPLERS._register_module, _wrap_register_module)
NNCF_PATCHER.patch(
    "mmdet.models.roi_heads.roi_extractors.SingleRoIExtractor.map_roi_levels",
    no_nncf_trace_wrapper,
)
NNCF_PATCHER.patch("mmdet.core.bbox2result", no_nncf_trace_wrapper)
NNCF_PATCHER.patch("mmdet.core.bbox2roi", no_nncf_trace_wrapper)


def _wrap_is_in_onnx_export(ctx, fn):
    # TODO: find a better way to solve this w/o patching 'torch.onnx.is_in_onnx_export'
    #
    # prevent incomplete graph building for MaskRCNN models
    #
    # possible alternatives
    #    - take the onnx branch in all cases

    import sys

    frame = sys._getframe()
    ctr = 2
    while frame is not None and ctr:
        frame = frame.f_back
        ctr -= 1
    if (
        frame is not None
        and frame.f_code.co_name == "forward"
        and "self" in frame.f_locals.keys()
        and frame.f_locals["self"].__class__.__name__ == "SingleRoIExtractor"
    ):
        return fn() or is_in_nncf_tracing()
    return fn()


NNCF_PATCHER.patch("torch.onnx.is_in_onnx_export", _wrap_is_in_onnx_export)

# add nncf context method that will be used when nncf tracing
BaseDetector.nncf_trace_context = nncf_trace_context


if is_mmdeploy_enabled():
    import mmdeploy.codebase.mmdet  # noqa: F401  # pylint: disable=unused-import
    from mmdeploy.core import FUNCTION_REWRITER
    from mmdeploy.core.rewriters.rewriter_utils import import_function

    for fn_path, record_dicts in FUNCTION_REWRITER._registry._rewrite_records.items():
        if fn_path.startswith("torch"):
            continue
        obj, obj_cls = import_function(fn_path)
        fn_name = fn_path.split(".")[-1]
        if _should_wrap(obj_cls, fn_name, HEADS_TARGETS):
            for record_dict in record_dicts:
                record_dict["_object"] = partial(no_nncf_trace_wrapper, None, record_dict["_object"])
