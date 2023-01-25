# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


class SAMClassifierMixin(object):
    """SAM-enabled BaseClassifier mix-in"""

    def train_step(self, data, optimizer=None, **kwargs):
        # Saving current batch data to compute SAM gradient
        # Rest of SAM logics are implented in SAMOptimizerHook
        self.current_batch = data

        return super().train_step(data, optimizer, **kwargs)
