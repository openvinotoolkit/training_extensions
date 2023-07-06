"""Detector Class for SAM optimizer."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


class SAMDetectorMixin:
    """SAM-enabled BaseDetector mix-in."""

    def train_step(self, data, optimizer, **kwargs):
        """Saving current batch data to compute SAM gradient."""
        # Rest of SAM logics are implented in SAMOptimizerHook
        self.current_batch = data

        return super().train_step(data, optimizer, **kwargs)
