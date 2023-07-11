"""Patch mmcls."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcls.models.classifiers.base import BaseClassifier

from otx.algorithms.common.adapters.nncf.patches import nncf_trace_context

# add nncf context method that will be used when nncf tracing
BaseClassifier.nncf_trace_context = nncf_trace_context
