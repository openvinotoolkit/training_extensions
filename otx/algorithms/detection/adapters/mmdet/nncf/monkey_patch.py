# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from functools import partial, partialmethod

from mmdet import core
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner
from mmdet.core.bbox.builder import BBOX_ASSIGNERS, BBOX_SAMPLERS
from mmdet.core.bbox.samplers import BaseSampler
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.dense_heads.base_mask_head import BaseMaskHead
from mmdet.models.detectors.base import BaseDetector
from mmdet.models.roi_heads.base_roi_head import BaseRoIHead
from mmdet.models.roi_heads.mask_heads.fcn_mask_head import FCNMaskHead
from mmdet.models.roi_heads.roi_extractors import SingleRoIExtractor

from otx.algorithms.common.adapters.nncf.monkey_patch import (
    conditioned_wrapper,
    nncf_trace_context,
    nncf_trace_wrapper,
    no_nncf_trace_wrapper,
)


#  from nncf.torch.dynamic_graph.context import get_current_context
#  if get_current_context() is not None and get_current_context().is_tracing:
#      __import__('ipdb').set_trace()


def wrap_mmdet_head(head_cls):
    TARGET_CLS = (BaseDenseHead, BaseMaskHead, BaseRoIHead, FCNMaskHead)
    TARGET_FNS = ("loss", "onnx_export", "get_bboxes", "get_seg_masks")

    conditioned_wrapper(head_cls, no_nncf_trace_wrapper, TARGET_FNS, TARGET_CLS)
    # 'onnx_export' method calls 'forward' method which need to be traced
    conditioned_wrapper(head_cls, nncf_trace_wrapper, ("forward",), TARGET_CLS)


def wrap_mmdet_bbox_assigner(bbox_assigner_cls):
    TARGET_CLS = (BaseAssigner,)
    TARGET_FNS = ("assign",)

    conditioned_wrapper(
        bbox_assigner_cls, no_nncf_trace_wrapper, TARGET_FNS, TARGET_CLS
    )


def wrap_mmdet_sampler(sampler_cls):
    TARGET_CLS = (BaseSampler,)
    TARGET_FNS = ("sample",)

    conditioned_wrapper(sampler_cls, no_nncf_trace_wrapper, TARGET_FNS, TARGET_CLS)


def wrap_register_module(self, *args, **kwargs):
    """
    A function to wrap classes lazily defined such as custom ones.
    """

    in_fn = kwargs.pop("in_fn")
    module = kwargs["module"]
    wrap_mmdet_head(module)
    wrap_mmdet_bbox_assigner(module)
    wrap_mmdet_sampler(module)
    return in_fn(*args, **kwargs)


# add nncf context method that will be used when nncf tracing
BaseDetector.nncf_trace_context = nncf_trace_context


# for mmdet defined heads
for head_cls in [BaseDenseHead, BaseMaskHead, BaseRoIHead] + list(
    HEADS.module_dict.values()
):
    wrap_mmdet_head(head_cls)


# for custom defined heads
HEADS._register_module = partialmethod(
    wrap_register_module, in_fn=HEADS._register_module
).__get__(HEADS)


# for mmdet defined bbox assigners
for bbox_assigner_cls in [BaseAssigner] + list(BBOX_ASSIGNERS.module_dict.values()):
    wrap_mmdet_bbox_assigner(bbox_assigner_cls)


# for custom defined bbox assigners
BBOX_ASSIGNERS._register_module = partialmethod(
    wrap_register_module, in_fn=BBOX_ASSIGNERS._register_module
).__get__(BBOX_ASSIGNERS)


# for mmdet defined samplers
# NNCF can not trace this part with torch older 1.11.0
for sampler_cls in [BaseSampler] + list(BBOX_SAMPLERS.module_dict.values()):
    wrap_mmdet_sampler(sampler_cls)


# for custom defined samplers
BBOX_SAMPLERS._register_module = partialmethod(
    wrap_register_module, in_fn=BBOX_SAMPLERS._register_module
).__get__(BBOX_SAMPLERS)


core.bbox2result = partial(no_nncf_trace_wrapper, in_fn=core.bbox2result)
core.bbox2roi = partial(no_nncf_trace_wrapper, in_fn=core.bbox2roi)


SingleRoIExtractor.map_roi_levels = partialmethod(
    no_nncf_trace_wrapper, in_fn=SingleRoIExtractor.map_roi_levels
)
