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
import torch
import torch.nn as nn
import torch.nn.functional as F
from .embedding import TransformerEmbedding
from .transformer_encoder import TransformerEncoder
from .transformer_cross_encoder import TransformerCrossEncoder
from .vae_bottleneck import VAEBottleneck


class LANMTEncoder(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            tgt_vocab_size,
            src_pad_idx,
            tgt_pad_idx,
            embed_size=512,
            hidden_size=512,
            n_att_heads=8,
            prior_layers=6,
            latent_dim=8,
            q_layers=6,
            # loss
            kl_budget=1.0,
            budget_annealing=False,
            max_steps=1
    ):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_att_heads = n_att_heads
        self.prior_layers = prior_layers
        self.latent_dim = latent_dim
        self.q_layers = q_layers
        # loss
        self.kl_budget = kl_budget
        self.budget_annealing = budget_annealing
        self.max_steps = max_steps
        self._init_modules()
        self._init_loss()

    def forward(self, src, src_mask, tgt=None, tgt_mask=None):
        out = {}
        out["prior_states"] = self.prior_encoder(src, src_mask)
        out["prior_prob"] = self.prior_prob_estimator(out["prior_states"])

        if tgt is None:
            out["latent"] = self.deterministic_sample_from_prob(out["prior_prob"])
        else:
            q_states = self.compute_Q_states(
                self.x_emb(src), src_mask, tgt, tgt_mask
            )
            out["sampled_z"], out["q_prob"] = self.sample_from_Q(q_states)

        return out

    def compute_Q_states(self, x_states, x_mask, y, y_mask):
        y_states = self.q_encoder_y(y, y_mask)
        if y.size(0) > x_states.size(0) and x_states.size(0) == 1:
            x_states = x_states.expand(y.size(0), -1, -1)
            x_mask = x_mask.expand(y.size(0), -1)
        states = self.q_encoder_xy(x_states, x_mask, y_states, y_mask)
        return states

    def sample_from_Q(self, q_states, sampling=True):
        sampled_z, q_prob = self.bottleneck(q_states, sampling=sampling)
        full_vector = self.latent2vector_nn(sampled_z)
        return full_vector, q_prob

    def deterministic_sample_from_prob(self, z_prob):
        mean_vector = z_prob[:, :, :self.latent_dim]
        full_vector = self.latent2vector_nn(mean_vector)
        return full_vector

    def _init_modules(self):
        # prior prob estimator p(z|x)
        self.x_emb = TransformerEmbedding(self.src_vocab_size, self.embed_size)
        self.prior_encoder = TransformerEncoder(
            self.x_emb, self.hidden_size, self.prior_layers, n_att_head=self.n_att_heads
        )
        self.prior_prob_estimator = nn.Linear(self.hidden_size, self.latent_dim * 2)

        # Approximator q(z|x,y)
        self.y_emb = TransformerEmbedding(self.tgt_vocab_size, self.embed_size)
        self.q_encoder_y = TransformerEncoder(
            self.y_emb, self.hidden_size, self.q_layers, n_att_head=self.n_att_heads
        )
        self.q_encoder_xy = TransformerCrossEncoder(None, self.hidden_size, self.q_layers)

        # Bottleneck
        self.bottleneck = VAEBottleneck(self.hidden_size, z_size=self.latent_dim)
        self.latent2vector_nn = nn.Linear(self.latent_dim, self.hidden_size)

    def _init_loss(self):
        self.loss = VAELoss(
            latent_dim = self.latent_dim,
            kl_budget = self.kl_budget,
            budget_annealing = self.budget_annealing,
            max_steps = self.max_steps
        )


class VAELoss(nn.Module):
    def __init__(self, latent_dim, kl_budget=1.0, budget_annealing=False, max_steps=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.kl_budget = kl_budget
        self.budget_annealing = budget_annealing
        self.max_steps = max_steps
        self.step = 0

    def forward(self, prior_prob, q_prob, src_mask):
        kl = self._compute_vae_kl(prior_prob, q_prob)

        # Apply budgets for KL divergence: KL = max(KL, budget)
        budget_upperbound = self.kl_budget
        budget = self._get_kl_budget()

        # Compute KL divergence
        max_mask = ((kl - budget) > 0.).float()
        kl = kl * max_mask + (1. - max_mask) * budget
        out = {"kl_loss": (kl * src_mask / src_mask.shape[0]).sum()}

        # report the averge KL for each token
        out["tok_kl"] = (kl * src_mask / src_mask.sum()).sum()

        return out

    def _compute_vae_kl(self, prior_prob, q_prob):
        mu1 = q_prob[:, :, :self.latent_dim]
        var1 = F.softplus(q_prob[:, :, self.latent_dim:])
        mu2 = prior_prob[:, :, :self.latent_dim]
        var2 = F.softplus(prior_prob[:, :, self.latent_dim:])
        kl = torch.log(var2 / (var1 + 1e-8) + 1e-8) + (
                    (torch.pow(var1, 2) + torch.pow(mu1 - mu2, 2)) / (2 * torch.pow(var2, 2))) - 0.5
        kl = kl.sum(-1)
        return kl

    def _get_kl_budget(self):
        if self.budget_annealing:
            self.step += 1
            half_maxsteps = float(self.max_steps / 2.0)
            if self.step > half_maxsteps:
                rate = (float(self.step) - half_maxsteps) / half_maxsteps
                min_budget = 0.
                budget = min_budget + (self.kl_budget - min_budget) * (1. - rate)
            else:
                budget = self.kl_budget
        else:
            budget = self.kl_budget
        return budget
