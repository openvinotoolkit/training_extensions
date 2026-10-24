# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom neck implementations for instance segmentation task."""

from .cspnext_pafpn import CSPNeXtPAFPN

__all__ = ["CSPNeXtPAFPN"]
