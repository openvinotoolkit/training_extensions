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

import torch
from torch import nn
from torch.nn import functional as F

from examples.object_detection.layers import DetectionOutput, PriorBox


class SSDDetectionOutput(nn.Module):
    def __init__(self, num_input_features, num_classes, config):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        self.heads = nn.ModuleList()
        for i, num_features in enumerate(num_input_features):
            self.heads.append(SSDHead(
                num_features, num_classes, config.min_sizes[i], config.max_sizes[i],
                config.aspect_ratios[i], config.steps[i], config.variance, config.flip, config.clip
            ))

        self.detection_output = DetectionOutput(
            num_classes, 0, config.get('top_k', 200), config.get('keep_top_k', 200), 0.01, 0.45, 1, 1,
            "CENTER_SIZE", 0
        )

    def forward(self, source_features, img_tensor):
        locs = []
        confs = []
        priors = []
        for features, head in zip(source_features, self.heads):
            loc, conf, prior = head(features, img_tensor)
            locs.append(loc)
            confs.append(conf)
            priors.append(prior)

        batch = source_features[0].size(0)
        loc = torch.cat([o.view(batch, -1) for o in locs], 1)
        conf = torch.cat([o.view(batch, -1) for o in confs], 1)
        conf_softmax = F.softmax(conf.view(conf.size(0), -1, self.num_classes), dim=-1)
        priors = torch.cat(priors, dim=2)

        if self.training:
            return loc.view(batch, -1, 4), conf.view(batch, -1, self.num_classes), priors.view(1, 2, -1, 4)
        return self.detection_output(loc, conf_softmax.view(batch, -1), priors)


class SSDHead(nn.Module):
    def __init__(self, num_features, num_classes, min_size, max_size, aspect_ratios, steps, varience, flip, clip):
        super().__init__()
        self.num_classes = num_classes
        self.clip = clip
        self.flip = flip
        self.varience = varience
        self.steps = steps
        self.aspect_ratios = aspect_ratios
        self.max_size = max_size
        self.min_size = min_size
        self.input_features = num_features

        num_prior_per_cell = 2 + 2 * len(aspect_ratios)
        self.loc = nn.Conv2d(num_features, num_prior_per_cell * 4, kernel_size=3, padding=1)
        self.conf = nn.Conv2d(num_features, num_prior_per_cell * num_classes, kernel_size=3, padding=1)
        self.prior_box = PriorBox(min_size, max_size, aspect_ratios, flip, clip, varience, steps, 0.5, 0, 0,
                                  0, 0, 0)

    def forward(self, features, image_tensor):
        loc = self.loc(features)
        conf = self.conf(features)
        priors = self.prior_box(features, image_tensor).to(loc.device)

        loc = loc.permute(0, 2, 3, 1).contiguous()
        conf = conf.permute(0, 2, 3, 1).contiguous()

        return loc, conf, priors


class MultiOutputSequential(nn.Sequential):
    def __init__(self, outputs, modules):
        super().__init__(*modules)
        self.outputs = [str(o) for o in outputs]

    def forward(self, x):
        outputs = []
        for name, module in self._modules.items():
            x = module(x)
            if name in self.outputs:
                outputs.append(x)
        return outputs, x
