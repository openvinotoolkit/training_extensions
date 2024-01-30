# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Warm-up schedulers for the OTX2.0."""

from lightning.pytorch.cli import ReduceLROnPlateau

class WarmupReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(self, optimizer, warmup_steps: int, monitor: str, warmup_by_epoch: bool = False, warmup_factor: float = 0.1, 
                 mode: str = 'min', factor: float = 0.1, patience: int = 10, threshold: float = 1e-4, 
                 threshold_mode: str = 'rel', cooldown : int = 0, min_lr: float = 0, eps: float = 1e-8, 
                 verbose: bool = False):
        
        self.warmup_steps = warmup_steps
        self.warmup_by_epoch = warmup_by_epoch
        super().__init__(optimizer, monitor, mode, factor, patience, threshold, threshold_mode,
                         cooldown, min_lr, eps, verbose)