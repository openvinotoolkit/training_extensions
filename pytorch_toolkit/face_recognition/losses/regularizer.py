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
import torch.nn.functional as F


def l2_reg_ortho(mdl):
    """
        Function used for Orthogonal Regularization.
    """
    l2_reg = None
    for w in mdl.parameters():
        if w.ndimension() < 2:
            continue
        else:
            cols = w[0].numel()
            w1 = w.view(-1, cols)
            wt = torch.transpose(w1, 0, 1)
            m = torch.matmul(wt, w1)
            ident = torch.eye(cols, cols).cuda()

            w_tmp = (m - ident)
            height = w_tmp.size(0)
            u = F.normalize(w_tmp.new_empty(height).normal_(0, 1), dim=0, eps=1e-12)
            v = F.normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
            u = F.normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
            sigma = torch.dot(u, torch.matmul(w_tmp, v))

            if l2_reg is None:
                l2_reg = (torch.norm(sigma, 2))**2
            else:
                l2_reg += (torch.norm(sigma, 2))**2
    return l2_reg


class ODecayScheduler():
    """Scheduler for the decay of the orthogonal regularizer"""
    def __init__(self, schedule, initial_decay, mult_factor):
        assert len(schedule) > 1
        self.schedule = schedule
        self.epoch_num = 0
        self.mult_factor = mult_factor
        self.decay = initial_decay

    def step(self):
        """Switches to the next step"""
        self.epoch_num += 1
        if self.epoch_num in self.schedule:
            self.decay *= self.mult_factor
        if self.epoch_num == self.schedule[-1]:
            self.decay = 0.0

    def get_decay(self):
        """Returns the current value of decay according to th schedule"""
        return self.decay
