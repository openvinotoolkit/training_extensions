"""
 Copyright (c) 2018 Intel Corporation
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
from losses.centroid_based import CenterLoss, PushLoss, MinimumMargin, PushPlusLoss, GlobalPushPlus


class MetricLosses:
    """Class-aggregator for all metric-learning losses"""
    def __init__(self, classes_num, embed_size, writer):
        self.writer = writer
        self.center_loss = CenterLoss(classes_num, embed_size, cos_dist=True)
        self.optimizer_centloss = torch.optim.SGD(self.center_loss.parameters(), lr=0.5)
        self.center_coeff = 0.0

        self.push_loss = PushLoss(soft=False, margin=0.7)
        self.push_loss_coeff = 0.0

        self.push_plus_loss = PushPlusLoss(margin=0.7)
        self.push_plus_loss_coeff = 0.0

        self.glob_push_plus_loss = GlobalPushPlus(margin=0.7)
        self.glob_push_plus_loss_coeff = 0.0

        self.min_margin_loss = MinimumMargin(margin=.7)
        self.min_margin_loss_coeff = 0.0

    def __call__(self, features, labels, epoch_num, iteration):
        log_string = ''

        center_loss_val = 0
        if self.center_coeff > 0.:
            center_loss_val = self.center_loss(features, labels)
            self.writer.add_scalar('Loss/center_loss', center_loss_val, iteration)
            log_string += ' Center loss: %.4f' % center_loss_val

        push_loss_val = 0
        if self.push_loss_coeff > 0.0:
            push_loss_val = self.push_loss(features, labels)
            self.writer.add_scalar('Loss/push_loss', push_loss_val, iteration)
            log_string += ' Push loss: %.4f' % push_loss_val

        push_plus_loss_val = 0
        if self.push_plus_loss_coeff > 0.0 and self.center_coeff > 0.0:
            push_plus_loss_val = self.push_plus_loss(features, self.center_loss.get_centers(), labels)
            self.writer.add_scalar('Loss/push_plus_loss', push_plus_loss_val, iteration)
            log_string += ' Push Plus loss: %.4f' % push_plus_loss_val

        glob_push_plus_loss_val = 0
        if self.glob_push_plus_loss_coeff > 0.0 and self.center_coeff > 0.0:
            glob_push_plus_loss_val = self.glob_push_plus_loss(features, self.center_loss.get_centers(), labels)
            self.writer.add_scalar('Loss/global_push_plus_loss', glob_push_plus_loss_val, iteration)
            log_string += ' Global Push Plus loss: %.4f' % glob_push_plus_loss_val

        min_margin_loss_val = 0
        if self.min_margin_loss_coeff > 0.0 and self.center_coeff > 0.0:
            min_margin_loss_val = self.min_margin_loss(self.center_loss.get_centers(), labels)
            self.writer.add_scalar('Loss/min_margin_loss', min_margin_loss_val, iteration)
            log_string += ' Min margin loss: %.4f' % min_margin_loss_val

        loss_value = self.center_coeff * center_loss_val + self.push_loss_coeff * push_loss_val + \
                     self.push_plus_loss_coeff * push_plus_loss_val + self.min_margin_loss_coeff * min_margin_loss_val \
                     + self.glob_push_plus_loss_coeff * glob_push_plus_loss_val

        if self.min_margin_loss_coeff + self.center_coeff + self.push_loss_coeff + self.push_plus_loss_coeff > 0.:
            self.writer.add_scalar('Loss/AUX_losses', loss_value, iteration)

        return loss_value, log_string

    def init_iteration(self):
        """Initializes a training iteration"""
        if self.center_coeff > 0.:
            self.optimizer_centloss.zero_grad()

    def end_iteration(self):
        """Finalizes a training iteration"""
        if self.center_coeff > 0.:
            for param in self.center_loss.parameters():
                param.grad.data *= (1. / self.center_coeff)
            self.optimizer_centloss.step()
