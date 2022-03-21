# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os
import string
import random

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from torchtts.models.attention import AttentionEncoder
from torchtts.text_preprocessing import text_to_sequence, symbols
from torchtts.models import commons
from torchtts.models.model import DurationPredictor, ResSequence, LayerNorm
from torchtts.models.model import ResStack2Stage, Residual2d, HighwayNetworkConv, ResSequenceDilated
import monotonic_align


class TextEncoder(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_chars,
                 prenet_k,
                 prenet_dims,
                 n_mels,
                 decoder_dims,
                 window_size=4,
                 dropout=0.5,
                 use_layer_norm=False):
        super().__init__()

        self.window_size = window_size
        self.embedding = nn.Embedding(num_chars, embed_dims)
        nn.init.normal_(self.embedding.weight, 0.0, embed_dims ** -0.5)

        self.dropout = dropout

        self.prenet = ResStack2Stage(embed_dims, prenet_dims, prenet_k, use_layer_norm)

        self.att_enc = AttentionEncoder(prenet_dims, 768,
                                        2, 6, 3, 0.1, window_size=window_size)

        self.proj_m = ResSequence(prenet_dims, n_mels, 5, use_layer_norm)
        self.proj_s = ResSequence(prenet_dims, decoder_dims, 5, use_layer_norm)
        self.proj_w = DurationPredictor(prenet_dims, filter_channels=64)

    def forward(self, x, x_lengths):
        x = self.embedding(x)

        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(1)), 1).to(x.dtype)

        x = x.transpose(1, 2)
        x = self.prenet(x, x_mask)

        x = self.att_enc(x, x_mask)

        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)  # mask by sequence length
        x_dp = torch.detach(x)

        x_m = self.proj_m(x, x_mask)

        x_res = self.proj_s(x, x_mask)

        logw = self.proj_w(x_dp, x_mask)

        return x_m, x_res, logw, x_mask

    def generate(self, x, x_lengths):
        x = self.embedding(x)

        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(1)), 1).to(x.dtype)

        x = x.transpose(1, 2)
        x = self.prenet(x, x_mask)

        x = self.att_enc(x, x_mask)

        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)  # mask by sequence length

        x_res = self.proj_s(x, x_mask)

        logw = self.proj_w(x, x_mask)

        return x_res, logw, x_mask

    def remove_weight_norm(self):
        self.prenet.remove_weight_norm()
        self.proj_m.remove_weight_norm()
        self.proj_s.remove_weight_norm()
        self.proj_w.remove_weight_norm()


class MelDecoder(nn.Module):
    def __init__(self,
                 postnet_k,
                 postnet_dims,
                 n_mels,
                 use_layer_norm=False
                 ):
        super().__init__()

        self.postnet1 = ResStack2Stage(postnet_dims, postnet_dims, postnet_k, use_layer_norm)

        self.proj2d = Residual2d(1, 3, 1, 8, batch_norm=False)

        self.postnet2 = ResSequenceDilated(postnet_dims, postnet_dims, postnet_k)

        self.post_proj = nn.Sequential(nn.Conv1d(postnet_dims, n_mels, 1),
                                       LayerNorm(n_mels),
                                       nn.PReLU(init=0.2),
                                       HighwayNetworkConv(n_mels),
                                       nn.Conv1d(n_mels, n_mels, 1),
                                       )

    def forward(self, x, mel_mask=None):
        if mel_mask is None:
            mel_lengths = torch.tensor([1, x.shape[2]]).long()
            mel_mask = torch.unsqueeze(commons.sequence_mask(mel_lengths, None), 1).to(x.dtype)

        x_post = x + self.postnet1(x, mel_mask)

        x_post = x_post + self.proj2d(x_post, mel_mask)

        x_post = x_post + self.postnet2(x_post, mel_mask)

        x_post = self.post_proj(x_post * mel_mask)

        return x_post * mel_mask

    def remove_weight_norm(self):
        self.postnet1.remove_weight_norm()
        self.postnet2.remove_weight_norm()
        self.proj2d1.remove_weight_norm()


class GANTacotron(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        extra_symbol = cfg["extra_symbol"] if "extra_symbol" in cfg else 0
        layer_norm = ("LayerNorm" in cfg and cfg["LayerNorm"] is True)

        if cfg.encoder.num_chars < len(symbols):
            cfg.encoder.num_chars = len(symbols)

        self.encoder = TextEncoder(cfg.encoder.embed_dims,
                                   cfg.encoder.num_chars + hasattr(cfg, "add_blank") + extra_symbol,
                                   cfg.encoder.prenet_k,
                                   cfg.encoder.prenet_dims,
                                   cfg.encoder.n_mels,
                                   cfg.decoder.postnet_dims,
                                   cfg.encoder.window_size,
                                   cfg.encoder.dropout,
                                   layer_norm)

        self.decoder = MelDecoder(cfg.decoder.postnet_k,
                                  cfg.decoder.postnet_dims,
                                  cfg.decoder.n_mels,
                                  layer_norm)

        self.register_buffer('step', torch.zeros(1, dtype=torch.long))

    def forward(self, x, x_length, mel, mel_lengths):
        x_m, x_res, logw, x_mask = self.encoder(x, x_length)

        mel_max_length = mel.size(2)
        mel, mel_lengths, mel_max_length = self.preprocess(mel, mel_lengths, mel_max_length)
        z_mask = torch.unsqueeze(commons.sequence_mask(mel_lengths, mel_max_length), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(z_mask, 2)

        with torch.no_grad():
            mul = torch.ones_like(x_m)  # [b, d, t]
            x_2 = torch.sum(-x_m ** 2, [1]).unsqueeze(-1)  # [b, t, 1]
            z_2 = torch.matmul(mul.transpose(1, 2), -mel ** 2)  # [b, t, t']
            xz2 = torch.matmul(x_m.transpose(1, 2), mel)  # [b, t, d] * [b, d, t'] = [b, t, t']

            corr_coeff = x_2 + z_2
            corr_coeff = corr_coeff + 2 * xz2

            attn = monotonic_align.maximum_path(corr_coeff, attn_mask.squeeze(1)).unsqueeze(1).detach()

        z_m = torch.matmul(attn.squeeze(1).transpose(1, 2), x_res.transpose(1, 2)).transpose(1,
                                                                                             2)  # [b, t', t], [b, t, d] -> [b, d, t']

        mel_proj = torch.matmul(attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)).transpose(1, 2)
        mel_proj = mel_proj + torch.randn_like(mel_proj)

        mel_proj_loss = torch.mean(torch.abs((mel_proj - mel) * z_mask))

        mel_ = self.decoder(z_m, z_mask)

        logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask
        return mel_, z_mask, (attn, logw, logw_, mel_proj_loss, mel_proj)

    def pad(self, x, max_len):
        x = x[:, :, :max_len]
        x = F.pad(x, [0, max_len - x.size(2), 0, 0], 'constant', 0.0)
        return x

    def preprocess(self, y, y_lengths, y_max_length):
        if y_max_length is not None:
            y = y[:, :, :y_max_length]
        return y, y_lengths, y_max_length

    def preprocessing_torch(self, x_res, log_dur, x_mask):
        w = torch.exp(log_dur) * x_mask
        w_ceil = torch.ceil(w)
        mel_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        mel_max_length = None

        _, mel_lengths, mel_max_length = self.preprocess(None, mel_lengths, mel_max_length)
        z_mask = torch.unsqueeze(commons.sequence_mask(mel_lengths, mel_max_length), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(z_mask, 2)

        attn = commons.generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
        z_m = torch.matmul(attn.squeeze(1).transpose(1, 2), x_res.transpose(1, 2)).transpose(1, 2)

        return z_m, z_mask, w_ceil

    @staticmethod
    def sequence_mask_numpy(length, max_length=None):
        if max_length is None:
            max_length = length.max()
        x = np.arange(max_length, dtype=length.dtype)
        return np.expand_dims(x, 0) < np.expand_dims(length, 1)

    @staticmethod
    def generate_path(duration, mask):
        """
        duration: [b, t_text]
        mask: [b, t_text, t_mel]
        """

        b, t_x, t_y = mask.shape  # batch size, text size, mel size
        cum_duration = np.cumsum(duration, 1)

        cum_duration_flat = cum_duration.flatten()
        path = GANTacotron.sequence_mask_numpy(cum_duration_flat, t_y).astype(mask.dtype)

        path = path.reshape(b, t_x, t_y)
        path = path - np.pad(path, ((0, 0), (1, 0), (0, 0)))[:, :-1]

        path = path * mask

        return path

    def preprocessing_numpy(self, x_res, log_dur, x_mask, offset=0.3):
        w = (np.exp(log_dur) + offset) * x_mask
        w_ceil = np.ceil(w)

        mel_lengths = np.clip(np.sum(w_ceil, axis=(1, 2)), a_min=1, a_max=None).astype(dtype=np.long)
        z_mask = np.expand_dims(self.sequence_mask_numpy(mel_lengths), 1).astype(x_mask.dtype)
        attn_mask = np.expand_dims(x_mask, -1) * np.expand_dims(z_mask, 2)

        attn = self.generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1))
        attn = np.expand_dims(attn, 1)
        attn = attn.squeeze(1).transpose(0, 2, 1)
        z_m = np.matmul(attn, x_res.transpose(0, 2, 1)).transpose(0, 2, 1)  # [b, t', t], [b, t, d] -> [b, d, t']

        return z_m, z_mask, w_ceil.squeeze()

    def to_onnx(self, dst_dir, cfg, text=None, target_text_len=128, target_mel_len=512,
                opset_version=11):
        if not os.path.isdir(dst_dir):
            os.mkdir(dst_dir)

        # 1) Convert encoder
        alphabet = string.ascii_lowercase
        if text is None or target_text_len > len(text):
            sz = len(alphabet) - 1
            text = "".join([alphabet[random.randint(0, sz)] for _ in range(target_text_len)])

        device = next(self.parameters()).device

        seq = text_to_sequence(text, cfg.data.text_cleaners)
        seq = torch.as_tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
        seq_len = torch.tensor([seq.shape[1]], dtype=torch.int32).to(device)

        print("ONNX conversion. Input shapes for encoder ", seq.shape, seq_len.shape)
        encoder_name = os.path.join(dst_dir, "encoder.onnx")
        print(encoder_name)
        torch.onnx.export(self.encoder, (seq, seq_len), encoder_name,
                          opset_version=opset_version,
                          do_constant_folding=True,
                          input_names=["seq", "seq_len"],
                          output_names=["x_m", "x_res", "log_dur", "x_mask"])

        # 2) Convert decoder
        x_m, x_res, log_dur, x_mask = self.encoder(seq, seq_len)

        z_res, z_mask, w_ceil = self.preprocessing_torch(x_res, log_dur, x_mask)

        # test numpy implementation
        z_res_np, z_mask_np, w_ceil_np = self.preprocessing_numpy(x_res.detach().cpu().numpy(),
                                                                  log_dur.detach().cpu().numpy(),
                                                                  x_mask.detach().cpu().numpy(),
                                                                  offset=0.0)

        def diff(t, n):
            return np.mean(np.abs(t.detach().cpu().numpy() - n))

        print(diff(z_res, z_res_np), diff(w_ceil, w_ceil_np), diff(z_mask, z_mask_np))
        z = z_res[:, :, :target_mel_len]
        z_mask = z_mask[:, :, :target_mel_len]

        print("ONNX conversion. Input shapes for decoder ", z.shape, z_mask.shape)
        decoder_name = os.path.join(dst_dir, "decoder.onnx")
        print(decoder_name)
        torch.onnx.export(self.decoder, (z, z_mask), decoder_name,
                          opset_version=opset_version,
                          do_constant_folding=True,
                          input_names=["z", "z_mask"],
                          output_names=["mel"])

        _ = self.decoder(z, z_mask)

        return encoder_name, decoder_name

    def generate(self, x, x_length):
        x_m, x_res, log_dur, x_mask = self.encoder(x, x_length)

        z_np, z_mask_np, mel_lengths = self.preprocessing_numpy(x_res.detach().cpu().numpy(),
                                                                log_dur.detach().cpu().numpy(),
                                                                x_mask.detach().cpu().numpy())
        z = torch.from_numpy(z_np).float()
        z_mask = torch.from_numpy(z_mask_np).float()

        y = self.decoder(z, z_mask)

        return y, x_m, mel_lengths

    def remove_weight_norm(self):
        self.encoder.remove_weight_norm()
        self.decoder.remove_weight_norm()
