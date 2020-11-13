"""
 Copyright (c) 2020 Intel Corporation
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
import torch.nn.functional as F


class VAEBottleneck(nn.Module):
    def __init__(self, hidden_size, z_size=None):
        super().__init__()
        self.hidden_size = hidden_size
        if z_size is None:
            self.z_size = self.hidden_size
        else:
            self.z_size = z_size
        self.dense = nn.Linear(hidden_size, self.z_size * 2)

    def forward(self, x, sampling=True, residual_q=None):
        vec = self.dense(x)
        mu = vec[:, :, :self.z_size]
        if residual_q is not None:
            mu = 0.5 * (mu + residual_q[:, :, :self.z_size])
        if not sampling:
            return mu, vec
        else:
            var = F.softplus(vec[:, :, self.z_size:])
            if residual_q is not None:
                var = 0.5 * (var + F.softplus(residual_q[:, :, self.z_size:]))
            noise = mu.clone()
            noise = noise.normal_()
            z = mu + noise * var
            return z, vec

    def sample_any_dist(self, dist, deterministic=False, samples=1, noise_level=1.):
        mu = dist[:, :, :self.z_size]
        if deterministic:
            return mu
        else:
            var = F.softplus(dist[:, :, self.z_size:])
            noise = mu.clone()
            if samples > 1:
                if noise.shape[0] == 1:
                    noise = noise.expand(samples, -1, -1).clone()
                    mu = mu.expand(samples, -1, -1).clone()
                    var = var.expand(samples, -1, -1).clone()
                else:
                    noise = noise[:, None, :, :].expand(-1, samples, -1, -1).clone()
                    mu = mu[:, None, :, :].expand(-1, samples, -1, -1).clone()
                    var = var[:, None, :, :].expand(-1, samples, -1, -1).clone()

            noise = noise.normal_()
            z = mu + noise * var * noise_level
            return z
