"""
This is a legacy file that serves to maintain compatibility with the OTE detection
framework. It links to the model_lifecycle enum under 'enums'

# TODO: Remove once https://jira.devtools.intel.com/browse/CVS-67869 is done
"""

# INTEL CONFIDENTIAL
#
# Copyright (C) 2021 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were provided to
# you ("License"). Unless the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit this software or the related documents
# without Intel's prior written permission.
#
# This software and the related documents are provided as is,
# with no express or implied warranties, other than those that are expressly stated
# in the License.
#

from .enums.model_lifecycle import ModelLifecycle

__all__ = ["ModelLifecycle"]
