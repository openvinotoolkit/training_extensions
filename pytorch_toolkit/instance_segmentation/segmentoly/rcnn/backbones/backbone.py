"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import torch.nn as nn

from ..base import FeatureExtractor


def freeze_params(module, freeze=True):
    for p in module.parameters():
        p.requires_grad = not freeze


def freeze_params_recursive(module, freeze=True, types=(nn.Module,)):
    for x in module.modules():
        if isinstance(x, types):
            freeze_params(x, freeze)


def freeze_mode(module, train_mode=False):
    module.train(train_mode)


def freeze_mode_recursive(module, train_mode=False, types=(nn.Module,)):
    for x in module.modules():
        if isinstance(x, types):
            freeze_mode(x, train_mode)


def freeze_params_and_mode(module, train_mode=False, freeze=True):
    module.train(train_mode)
    freeze_params(module, freeze)


def freeze_params_and_mode_recursive(module, train_mode=False, freeze=True, types=(nn.Module,)):
    for x in module.modules():
        if isinstance(x, types):
            freeze_params_and_mode(x, train_mode, freeze)


class Backbone(FeatureExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.dims_in = (3,)
        self.scales_in = (1,)
        self.stages = nn.Sequential()
        self._all_dims_out = tuple(1 for _ in range(len(self.stages)))
        self._all_scales_out = tuple(2 ** (i + 1) for i in range(len(self.stages)))
        self.output_stages = list(range(len(self.stages)))
        self.dims_out = list(self._all_dims_out[i] for i in self.output_stages)
        self.scales_out = list(self._all_scales_out[i] for i in self.output_stages)

        self.stages_with_frozen_params = ()
        self.stages_with_frozen_mode = ()
        self.batch_norm_types = (nn.BatchNorm2d, )

    def set_output_stages(self, output_stages):
        self.output_stages = output_stages
        self.dims_out = list(self._all_dims_out[i] for i in self.output_stages)
        self.scales_out = list(self._all_scales_out[i] for i in self.output_stages)

    def freeze_stages_params(self, stage_indices):
        self.stages_with_frozen_params = stage_indices
        # Unfreeze whole model first.
        freeze_params_recursive(self, False)
        # Freeze only relevant stages.
        for stage_idx in stage_indices:
            freeze_params(self.stages[stage_idx], True)

    def freeze_stages_bns(self, stage_indices):
        self.stages_with_frozen_mode = stage_indices
        for stage_idx in range(len(self.stages)):
            freeze_params_recursive(self.stages[stage_idx], stage_idx in stage_indices, self.batch_norm_types)
        self.train(self.training)

    def train(self, train_mode=True):
        super().train(train_mode)
        self.training = train_mode
        for stage_idx, stage in enumerate(self.stages):
            freeze_mode_recursive(stage, train_mode if stage_idx not in self.stages_with_frozen_mode else False,
                                  types=self.batch_norm_types)

    def forward(self, x):
        src_width = x.shape[-1]
        outputs_per_stage = []
        for stage_idx, stage in enumerate(self.stages):
            x = stage(x)
            outputs_per_stage.append(x)
        outputs = list(outputs_per_stage[i] for i in self.output_stages)
        for output, scale in zip(outputs, self.scales_out):
            assert output.shape[-1] * scale == src_width, '{} * {} != {}'.format(x.shape[-1], scale, src_width)
        return outputs
