"""This is a legacy file that serves to maintain compatibility with the OTX detection framework.

It links to the model_lifecycle enum under 'enums'

# TODO: Remove once https://jira.devtools.intel.com/browse/CVS-67869 is done
"""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .enums.model_lifecycle import ModelLifecycle

__all__ = ["ModelLifecycle"]
