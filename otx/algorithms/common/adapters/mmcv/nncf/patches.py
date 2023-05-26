"""Patch mmcv and mpa stuff."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from copy import deepcopy

from otx.algorithms.common.adapters.nncf import (
    NNCF_PATCHER,
    no_nncf_trace_wrapper,
)


# pylint: disable-next=unused-argument,invalid-name
def _evaluation_wrapper(self, fn, runner, *args, **kwargs):
    # TODO: move this patch to upper level (mmcv)
    # as this is not only nncf required feature.
    # one example is ReduceLROnPlateauLrUpdaterHook
    out = fn(runner, *args, **kwargs)
    setattr(runner, "all_metrics", deepcopy(runner.log_buffer.output))
    return out


NNCF_PATCHER.patch("mmcv.runner.EvalHook.evaluate", _evaluation_wrapper)
NNCF_PATCHER.patch("otx.algorithms.common.adapters.mmcv.hooks.eval_hook.CustomEvalHook.evaluate", _evaluation_wrapper)

NNCF_PATCHER.patch(
    "otx.algorithms.common.adapters.mmcv.hooks.recording_forward_hook.FeatureVectorHook.func",
    no_nncf_trace_wrapper,
)
NNCF_PATCHER.patch(
    "otx.algorithms.common.adapters.mmcv.hooks.recording_forward_hook.ActivationMapHook.func",
    no_nncf_trace_wrapper,
)
NNCF_PATCHER.patch(
    "otx.algorithms.common.adapters.mmcv.hooks.recording_forward_hook.ReciproCAMHook.func",
    no_nncf_trace_wrapper,
)
