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
from .modules.lanmt_encoder import LANMTEncoder
from .modules.length_predictor import LengthPredictor
from .modules.mask_generator import MaskGenerator
from .modules.length_converter import LengthConverter
from .modules.transformer_cross_encoder import TransformerCrossEncoder
from .modules.token_prediction_loss import TokenPredictionLoss

class LANMT(nn.Module):
    def __init__(
            self,
            # encoder
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
            # length predictor
            max_delta=50,
            # decoder
            decoder_layers=6,
            # loss params
            kl_budget=1.0,
            budget_annealing=False,
            max_steps=1,
            **kwargs

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
        self.max_delta = max_delta
        self.decoder_layers = decoder_layers
        self.kl_budget = kl_budget
        self.budget_annealing = budget_annealing
        self.max_steps = max_steps
        self._init_modules()
        self._init_loss()

    def forward(self, src, tgt=None, max_len=None, **kwargs):
        src_mask = torch.ne(src, self.src_pad_idx).float()
        tgt_mask = None if tgt is None else torch.ne(tgt, self.tgt_pad_idx).float()

        # encoder
        out = self.encoder(src, src_mask, tgt, tgt_mask)
        latent = out["latent"] if tgt is None else out["sampled_z"]

        # predict length
        out["length_predictor_logits"], delta = self.length_predictor(
            out["prior_states"] + latent, src_mask
        )

        # generate tgt_mask & tgt_lens
        if tgt is not None:
            tgt_lens, out["tgt_mask"] = tgt_mask.sum(-1).long(), tgt_mask
        else:
            tgt_lens, out["tgt_mask"] = self.mask_generator(
                src_mask, delta + 1, max_len
            )

        # convert src embeddings to tgt_lens
        pred_emb = self.length_converter(latent, src_mask, tgt_lens, max_len)

        # decoder
        decoder_states = self.decoder(
            pred_emb, out["tgt_mask"], out["prior_states"], src_mask
        )
        out["token_predictor_logits"] = self.token_predictor(decoder_states)
        return out

    def loss(
            self,
            length_predictor_logits,
            token_predictor_logits,
            prior_prob, q_prob,
            src, tgt, **kwargs
    ):
        out = {}
        src_mask = torch.ne(src, self.src_pad_idx).float()
        tgt_mask = torch.ne(tgt, self.tgt_pad_idx).float()
        # length_prediction_loss
        out.update(self.length_predictor.loss(length_predictor_logits, src_mask, tgt_mask))
        # encoder loss
        out.update(self.encoder.loss(prior_prob, q_prob, src_mask))
        # token_prediction_loss
        out.update(self.token_prediction_loss(token_predictor_logits, tgt, tgt_mask))
        # loss
        loss = 0
        for k, v in out.items():
            if "_loss" in k:
                loss += v
        out["loss"] = loss
        return out

    def get_params(self):
        params = {"params": self.parameters(), "lr": 1.0}
        return [params]

    def _init_modules(self):
        # encoder
        self.encoder = LANMTEncoder(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            src_pad_idx=self.src_pad_idx,
            tgt_pad_idx=self.tgt_pad_idx,
            embed_size=self.embed_size,
            hidden_size=self.hidden_size,
            n_att_heads=self.n_att_heads,
            prior_layers=self.prior_layers,
            latent_dim=self.latent_dim,
            q_layers=self.q_layers,
            # loss
            kl_budget=self.kl_budget,
            budget_annealing=self.budget_annealing,
            max_steps=self.max_steps
        )
        # length predictor
        self.length_predictor = LengthPredictor(self.hidden_size, self.max_delta)
        # tgt_mask generator
        self.mask_generator = MaskGenerator()
        # length converter
        self.length_converter = LengthConverter(self.hidden_size)
        # decoder
        self.decoder = TransformerCrossEncoder(
            None, self.hidden_size, self.decoder_layers, skip_connect=True
        )
        # token predictor
        self.token_predictor = nn.Linear(self.hidden_size, self.tgt_vocab_size)

    def _init_loss(self):
        self.token_prediction_loss = TokenPredictionLoss(
            tgt_vocab_size=self.tgt_vocab_size,
            ignore_first_token=False
        )
