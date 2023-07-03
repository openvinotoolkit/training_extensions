"""Tile Samipling Hook."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from random import sample

from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class TileSamplingHook(Hook):
    """Tile Sampling Hook.

    Usually training model with tile requires lots of time due to large images generate plenty of tiles.
    To save training and validation time, OTX offers samipling method to entire tile datset.
    Especially tile sampling hook samples tiles whenever epoch starts to train model various tile samples
    """

    def before_epoch(self, runner):
        """Sample tiles from training datset when epoch starts."""
        if hasattr(runner.data_loader.dataset, "tile_dataset"):
            tile_dataset = runner.data_loader.dataset.tile_dataset
            tile_dataset.tiles = sample(tile_dataset.tiles_all, tile_dataset.sample_num)
