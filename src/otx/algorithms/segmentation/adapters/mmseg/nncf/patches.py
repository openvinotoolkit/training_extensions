"""Patch mmseg library."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# false positive pylint: disable=no-name-in-module
from mmseg.models.segmentors.base import BaseSegmentor

from otx.algorithms.common.adapters.nncf.patches import nncf_trace_context

# add nncf context method that will be used when nncf tracing
BaseSegmentor.nncf_trace_context = nncf_trace_context
