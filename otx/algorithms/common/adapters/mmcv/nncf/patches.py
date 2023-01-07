"""Patch mmcv and mpa stuff."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from copy import deepcopy

from otx.algorithms.common.adapters.nncf.patchers import (
    NNCF_PATCHER,
    no_nncf_trace_wrapper,
)
from otx.algorithms.common.adapters.nncf.utils import is_nncf_enabled

if is_nncf_enabled():
    from nncf.torch.nncf_network import NNCFNetwork

    # pylint: disable-next=ungrouped-imports
    from otx.algorithms.common.adapters.nncf.patches import nncf_train_step

    # add wrapper train_step method
    NNCFNetwork.train_step = nncf_train_step


# pylint: disable-next=unused-argument,invalid-name
def _evaluation_wrapper(self, fn, runner, *args, **kwargs):
    out = fn(runner, *args, **kwargs)
    setattr(runner, "all_metrics", deepcopy(runner.log_buffer.output))
    return out


NNCF_PATCHER.patch("mmcv.runner.EvalHook.evaluate", _evaluation_wrapper)
NNCF_PATCHER.patch("otx.mpa.modules.hooks.eval_hook.CustomEvalHook.evaluate", _evaluation_wrapper)

NNCF_PATCHER.patch(
    "otx.mpa.modules.hooks.recording_forward_hooks.FeatureVectorHook.func",
    no_nncf_trace_wrapper,
)
NNCF_PATCHER.patch(
    "otx.mpa.modules.hooks.recording_forward_hooks.ActivationMapHook.func",
    no_nncf_trace_wrapper,
)
NNCF_PATCHER.patch(
    "otx.mpa.modules.hooks.recording_forward_hooks.ReciproCAMHook.func",
    no_nncf_trace_wrapper,
)
