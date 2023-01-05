"""Adapters for nncf support."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# flake8: noqa

from .compression import (
    AccuracyAwareLrUpdater,
    get_nncf_metadata,
    get_uncompressed_model,
    is_checkpoint_nncf,
    is_state_nncf,
)
from .patches import *
from .utils import (
    check_nncf_is_enabled,
    get_nncf_version,
    is_accuracy_aware_training_set,
    is_in_nncf_tracing,
    no_nncf_trace,
)

__all__ = [
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
]
