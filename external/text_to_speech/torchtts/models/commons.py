# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Based on Glow-TTS https://github.com/jaywalnut310/glow-tts
# Copyright (c) 2020 Jaehyeon Kim
# SPDX-License-Identifier: MIT
#
# Glow-TTS contains snippet from WaveGlow https://github.com/NVIDIA/waveglow
# Copyright (c) 2018, NVIDIA Corporation
# SPDX-License-Identifier: BSD-3-Clause
#
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


# From WaveGlow
@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def shift_1d(x):
    x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
    return x


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def maximum_path(value, mask, max_neg_val=-np.inf):
    """ Numpy-friendly version. It's about 4 times faster than torch version.
    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]
    """
    value = value * mask

    device = value.device
    dtype = value.dtype
    value = value.cpu().detach().numpy()
    mask = mask.cpu().detach().numpy().astype(np.bool)

    b, t_x, t_y = value.shape
    direction = np.zeros(value.shape, dtype=np.int64)
    v = np.zeros((b, t_x), dtype=np.float32)
    x_range = np.arange(t_x, dtype=np.float32).reshape(1, -1)
    for j in range(t_y):
        v0 = np.pad(v, [[0, 0], [1, 0]], mode="constant", constant_values=max_neg_val)[:, :-1]
        v1 = v
        max_mask = (v1 >= v0)
        v_max = np.where(max_mask, v1, v0)
        direction[:, :, j] = max_mask

        index_mask = (x_range <= j)
        v = np.where(index_mask, v_max + value[:, :, j], max_neg_val)
    direction = np.where(mask, direction, 1)

    path = np.zeros(value.shape, dtype=np.float32)
    index = mask[:, :, 0].sum(1).astype(np.int64) - 1
    index_range = np.arange(b)
    for j in reversed(range(t_y)):
        path[index_range, index, j] = 1
        index = index + direction[index_range, index, j] - 1
    path = path * mask.astype(np.float32)
    path = torch.from_numpy(path).to(device=device, dtype=dtype)
    return path


def generate_path(duration, mask):
    """
    duration: [b, t_x]
    mask: [b, t_x, t_y]
    """
    device = duration.device

    b, t_x, t_y = mask.shape  # batch size, text size, mel size
    cum_duration = torch.cumsum(duration, 1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)

    path = path.view(b, t_x, t_y)
    cps = convert_pad_shape([[0, 0], [1, 0], [0, 0]])
    path = path - F.pad(path, cps)[:, :-1]

    path = path * mask

    return path


class Adam():
    def __init__(self, params, dim_model, scheduler="noam", warmup_steps=4000, lr=1e0, betas=(0.9, 0.98), eps=1e-9):
        self.params = params
        self.scheduler = scheduler
        self.dim_model = dim_model
        self.warmup_steps = warmup_steps
        self.lr = lr
        self.betas = betas
        self.eps = eps

        self.step_num = 1
        self.cur_lr = lr * self._get_lr_scale()

        self._optim = torch.optim.Adam(params, lr=self.cur_lr, betas=betas, eps=eps)

    def _get_lr_scale(self):
        if self.scheduler == "noam":
            return np.power(self.dim_model, -0.5) * np.min(
                [np.power(self.step_num, -0.5), self.step_num * np.power(self.warmup_steps, -1.5)])
        else:
            return 1

    def _update_learning_rate(self):
        self.step_num += 1
        if self.scheduler == "noam":
            self.cur_lr = self.lr * self._get_lr_scale()
            for param_group in self._optim.param_groups:
                param_group['lr'] = self.cur_lr

    def get_lr(self):
        return self.cur_lr

    def step(self):
        self._optim.step()
        self._update_learning_rate()

    def zero_grad(self):
        self._optim.zero_grad()

    def load_state_dict(self, d):
        self._optim.load_state_dict(d)

    def state_dict(self):
        return self._optim.state_dict()


def clip_grad_value_(parameters, clip_value, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type

        p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def squeeze(x, x_mask=None, n_sqz=2):
    b, c, t = x.size()

    t = (t // n_sqz) * n_sqz
    x = x[:, :, :t]
    x_sqz = x.view(b, c, t // n_sqz, n_sqz)
    x_sqz = x_sqz.permute(0, 3, 1, 2).contiguous().view(b, c * n_sqz, t // n_sqz)

    if x_mask is not None:
        x_mask = x_mask[:, :, n_sqz - 1::n_sqz]
    else:
        x_mask = torch.ones(b, 1, t // n_sqz).to(device=x.device, dtype=x.dtype)
    return x_sqz * x_mask, x_mask


def unsqueeze(x, x_mask=None, n_sqz=2):
    b, c, t = x.size()

    x_unsqz = x.view(b, n_sqz, c // n_sqz, t)
    x_unsqz = x_unsqz.permute(0, 2, 3, 1).contiguous().view(b, c // n_sqz, t * n_sqz)

    if x_mask is not None:
        x_mask = x_mask.unsqueeze(-1).repeat(1, 1, 1, n_sqz).view(b, 1, t * n_sqz)
    else:
        x_mask = torch.ones(b, 1, t * n_sqz).to(device=x.device, dtype=x.dtype)
    return x_unsqz * x_mask, x_mask
