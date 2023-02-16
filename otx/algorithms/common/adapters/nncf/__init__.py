"""Adapters for nncf support."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# flake8: noqa

from otx.core.patcher import Patcher

from .compression import (
    AccuracyAwareLrUpdater,
    get_nncf_metadata,
    get_uncompressed_model,
    is_checkpoint_nncf,
    is_state_nncf,
)
from .patches import (
    nncf_trace_context,
    nncf_trace_wrapper,
    nncf_train_step,
    no_nncf_trace_wrapper,
)
from .utils import (
    check_nncf_is_enabled,
    get_nncf_version,
    is_accuracy_aware_training_set,
    is_in_nncf_tracing,
    is_nncf_enabled,
    no_nncf_trace,
)

NNCF_PATCHER = Patcher()


__all__ = [
    "NNCF_PATCHER",
    "AccuracyAwareLrUpdater",
    "check_nncf_is_enabled",
    "get_nncf_metadata",
    "get_nncf_version",
    "get_uncompressed_model",
    "is_accuracy_aware_training_set",
    "is_checkpoint_nncf",
    "is_in_nncf_tracing",
    "is_state_nncf",
    "no_nncf_trace",
    "nncf_trace_context",
    "nncf_train_step",
    "no_nncf_trace_wrapper",
    "nncf_trace_wrapper",
    "is_nncf_enabled",
]
