# Copyright (c) 2018, Sergei Belousov
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# The original repo: https://github.com/bes-dev/mpl.pytorch

import torch
from mmseg.core.seg.builder import PIXEL_SAMPLERS
from mmseg.core.seg.sampler.base_pixel_sampler import BasePixelSampler

from ...utils.ext_loader import load_ext

ext_module = load_ext("otx.mpa.modules._mpl", ["compute_weights"])


@PIXEL_SAMPLERS.register_module()
class MaxPoolingPixelSampler(BasePixelSampler):
    """Max-Pooling Loss
    Implementation of "Loss Max-Pooling for Semantic Image Segmentation"
    https://arxiv.org/abs/1704.02966
    """

    def __init__(self, ratio=0.3, p=1.7, skip_max_ratio=None, **kwargs):
        super().__init__(**kwargs)

        assert 0 < ratio <= 1, "ratio should be in range [0, 1]"
        assert p > 1, "p should be > 1"

        self.ratio = ratio
        self.p = p

        self.skip_max_ratio = skip_max_ratio
        if self.skip_max_ratio is not None:
            assert 0.0 < self.skip_max_ratio < 1.0

    def sample(self, seg_logit=None, seg_label=None, losses=None, valid_mask=None):
        assert losses is not None

        with torch.no_grad():
            if valid_mask is None:
                assert seg_label is not None
                valid_mask = seg_label != self.ignore_index

            flat_losses = losses.view(-1)
            valid_losses = flat_losses[valid_mask.view(-1)]

            if self.skip_max_ratio is not None:
                max_skipped = max(1, int(valid_losses.size(0) * self.skip_max_ratio))
                assert max_skipped < valid_losses.size(0)

                num_skipped = torch.randint(1, max_skipped + 1, [])
                _sort_losses, _ = valid_losses.sort()
                ignore_threshold = _sort_losses[-num_skipped]
                valid_losses = torch.where(
                    valid_losses > ignore_threshold,
                    torch.zeros_like(valid_losses),
                    valid_losses,
                )

            sort_losses, sort_indices = valid_losses.sort()
            sort_losses = sort_losses.contiguous()
            sort_indices = sort_indices.contiguous()

            weights = torch.zeros(sort_losses.size()).contiguous()
            ext_module.compute_weights(
                sort_losses.size(0),
                sort_losses.cpu(),
                sort_indices.cpu(),
                weights.cpu(),
                self.ratio,
                self.p,
            )

            seg_weight = torch.zeros_like(losses)
            seg_weight[valid_mask] = float(sort_losses.size(0)) * weights.to(losses.device)

            return seg_weight
