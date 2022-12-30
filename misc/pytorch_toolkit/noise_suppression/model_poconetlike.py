"""
 Copyright (c) 2021 Intel Corporation

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

import logging
import os
import math
import re

import numpy as np
import torch

from model_base import BaseDNSModel
from states import States
from utils import get_shape

logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',level=logging.INFO)
logger = logging.getLogger('{} model_poconetlike'.format(os.getpid()))
def printlog(*args):
    logger.info(' '.join([str(v) for v in args]))

class MHA(torch.nn.Module):
    def __init__(self, attn_range, channels, out_channels, n_heads, attn_unroll=128):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.attn_range = attn_range
        self.attn_unroll = attn_unroll

        self.k_channels = channels // n_heads
        self.linear_qkv = torch.nn.Conv2d(channels, channels * 3, (1,1))
        torch.nn.init.xavier_uniform_(self.linear_qkv.weight)

        self.conv_o = torch.nn.Conv2d(channels, out_channels, (1,1))

    def forward(self, x, state_old=None):

        B, C, F, _ = get_shape(x)

        # [B,C,F,T] -> [B,3C,F,T]
        qkv = self.linear_qkv(x)

        # [B,3C,F,T] -> [B,F,3C,T]
        qkv = qkv.transpose(1, 2)

        # [B,F,3C,T] -> 3*[B,F,C,T]
        query, key, value = qkv.chunk(3, dim=2)

        #[B,F,C,T] -> [B,F,C,attn_range+T]
        states = States(state_old)
        key = states.pad_left(key, self.attn_range[0], dim=3)
        value = states.pad_left(value, self.attn_range[0], dim=3)

        # [B, F, C, T] -> [BF, n_heads, k_channels, T]
        T_q = get_shape(query)[3]
        T_kv = get_shape(key)[3]
        assert C == self.n_heads*self.k_channels
        query = query.reshape(B*F, self.n_heads, self.k_channels, T_q)
        key = key.reshape(B*F, self.n_heads, self.k_channels, T_kv)
        value = value.reshape(B*F, self.n_heads, self.k_channels, T_kv)

        # transpose [BF, n_heads, k_channels, T] -> [BF, n_heads, T, k_channels]
        query = query.transpose(2, 3)
        key = key.transpose(2, 3)
        value = value.transpose(2, 3)

        assert self.attn_unroll > 0
        q_s = 0
        outputs = []
        while q_s < T_q:
            q_e = min(q_s + self.attn_unroll, T_q)
            kv_s = q_s
            kv_e = min(q_e + sum(self.attn_range), T_kv)
            if q_s == 0 and q_e == T_q:
                query_slice = query
            else:
                query_slice = query[:, :, q_s:q_e, :]
            if kv_s == 0 and kv_e == T_kv:
                key_slice = key
                value_slice = value
            else:
                key_slice = key[:, :, kv_s:kv_e, :]
                value_slice = value[:, :, kv_s:kv_e, :]
            scores = torch.matmul(query_slice, key_slice.transpose(-2, -1)) / math.sqrt(self.k_channels)

            # mask key scores that out of attn range
            q_n = q_e - q_s
            k_mn = 1 + sum(self.attn_range)
            mask = torch.ones(q_n, k_mn, dtype=scores.dtype, device=scores.device)
            mask = torch.nn.functional.pad(mask, (0, q_n), mode='constant', value=0)
            mask = mask.reshape(-1)[:q_n * (k_mn + q_n - 1)]
            mask = mask.reshape(q_n, (k_mn + q_n - 1))
            mask = mask[:, :kv_e - kv_s]
            mask = mask.unsqueeze(0).unsqueeze(0)

            scores = scores * mask + -1e4 * (1 - mask)
            p_attn = torch.nn.functional.softmax(scores, dim=3)  # [b, n_h, l_q, l_kv]

            output = torch.matmul(p_attn, value_slice)
            outputs.append(output)
            q_s = q_e

        output = torch.cat(outputs, 2) if len(outputs) > 1 else outputs[0]

        # [BF, n_h, T_q, d_k] -> [BF, n_h, d_k, T_q]
        output = output.transpose(2, 3)

        # [BF, n_h, d_k, T_q] -> [B, F, C, T_q]
        output = output.reshape(B, F, C, T_q)

        # [B, F, C, T] -> [B, C, F, T]
        output = output.transpose(1, 2)

        #[B, C, F, T] -> [B, Co, F, T]
        output = self.conv_o(output)

        return output, states.state


class BlockX(torch.nn.Module):
    def __init__(self, ks, pad, inp_ch, out_ch):
        super().__init__()

        self.bn = torch.nn.BatchNorm2d(inp_ch)
        self.act = torch.nn.LeakyReLU()
        self.pad = pad
        self.conv = torch.nn.Conv2d(inp_ch, out_ch, kernel_size=ks)

    def forward(self, x, state_old):
        states = States(state_old)
        x = self.bn(x)
        x = self.act(x)
        if self.pad is not None:
            #x = self.pad(x)
            pad = self.pad
            if pad[0] > 0:
                #save last times to use on next iter for left padding
                x = states.pad_left(x, pad[0], 3)
                pad = (0,) + pad[1:]
            x = torch.nn.functional.pad(x, pad, mode='constant', value=0)

        x = self.conv(x)
        return x, states.state

class BlockDenseMHA(torch.nn.Module):
    def __init__(self, inp_ch, out_ch, grow, mha_heads_num, mha_range, back_time_flag):
        super().__init__()
        self.inp_ch = inp_ch
        self.out_ch = out_ch
        self.mha_heads_num = mha_heads_num
        dense_steps = 2

        ks = (3,3)

        #symmetrical padding for freq
        assert ks[0] % 2 == 1
        pad_f = (ks[0] // 2, ks[0] // 2)
        #padding for time in past
        if back_time_flag:
            pad_t = (ks[1] - 1, 0 )# no future
        else:
            pad_t = (ks[1] // 2, ks[1] - 1 - ks[1] // 2)  # with future
        pad = pad_t + pad_f

        bd_list = [BlockX(ks, pad, inp_ch + grow*i, grow) for i in range(dense_steps)]
        self.bd = torch.nn.ModuleList(bd_list)

        self.bd_out = BlockX(ks, pad, inp_ch+dense_steps*grow, out_ch)

        self.sa = None
        if mha_range>0:
            if back_time_flag:
                block_length = (mha_range, 0)
            else:
                block_length = (mha_range, mha_range)

            self.sa = MHA(
                block_length,
                out_ch,
                out_ch,
                mha_heads_num
            )

            self.sa_out = BlockX((1, 1), None, out_ch*2, out_ch)

    def forward(self, x, state_old):
        states = States(state_old)

        for m in self.bd:
            x_add, state = m(x, states.state_old)
            states.update(state)
            x = torch.cat([x,x_add],1)
        x, state = self.bd_out(x, states.state_old)
        states.update(state)

        if self.sa is None:
            return x, states.state

        #apply multihead self attension module
        #[B,C,F,T]
        x_add,state = self.sa(x, state_old=states.state_old)
        states.update(state)

        x = torch.cat([x,x_add],1)

        x,state = self.sa_out(x,states.state_old)
        states.update(state)

        return x, states.state

class PoCoNetLikeModel(BaseDNSModel):
    def __init__(self, **kwargs):

        kwargs = kwargs.copy()

        def kwargs_from_desc(name, name_desc, type_conv, default_val):
            if name not in kwargs:
                val = [default_val]
                if 'model_desc' in kwargs:
                    m = re.match(r'.*{}([\d\.\:]+).*'.format(name_desc), kwargs['model_desc'])
                    if m:
                        val = [type_conv(v) for v in m.group(1).split(":")]
                kwargs[name] = val if len(val)>1 else val[0]

        kwargs_from_desc('wnd_length',     'wnd',  int, 256)
        kwargs_from_desc('hop_length',     'hop',  int, kwargs['wnd_length']//2)
        kwargs_from_desc('ahead',          'ahead',int, 4)
        kwargs_from_desc('back_time_flag', 'bt',   int, 1)
        kwargs_from_desc('mha_ranges',     'mhar', int, [32, 22, 16, 11, 8])

        super().__init__(**kwargs)

        self.wnd_length = kwargs['wnd_length']
        self.hop_length = kwargs['hop_length']
        self.ahead = kwargs['ahead']
        self.back_time_flag = kwargs['back_time_flag']
        self.mha_ranges = kwargs['mha_ranges']

        #samples that are not covered all fft windows removed.
        self.ahead_ifft = (self.wnd_length - self.hop_length) // self.hop_length

        #init encoder and decoder weights
        t = np.eye(self.wnd_length)
        fft_matrix = np.fft.rfft(t).transpose()

        #normalize fft_matrix to be ortonormal
        n = np.linalg.norm(fft_matrix, axis=1, keepdims=True)
        n[1:-1] *= np.sqrt(0.5) #reduce norm for cos,sin pair and keep cos only
        fft_matrix /= n

        fft_matrix = np.vstack([
            np.real(fft_matrix),
            np.imag(fft_matrix)
        ])

        window = np.hanning(self.wnd_length + 1)[:-1]
        window = window * 2 / (self.wnd_length // self.hop_length)

        conv_matrix = fft_matrix * (window[None,:]**0.5)
        conv_matrix = torch.from_numpy(conv_matrix).unsqueeze(1).unsqueeze(2).float()
        self.register_buffer("_encdec_matrix", conv_matrix)

        self.pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.blocks_down = torch.nn.ModuleList([
            BlockDenseMHA(2, 32, 16,
                          mha_heads_num=1,
                          mha_range=self.mha_ranges[0],
                          back_time_flag=self.back_time_flag),
            BlockDenseMHA(32, 64, 16,
                          mha_heads_num=2,
                          mha_range=self.mha_ranges[1],
                          back_time_flag=self.back_time_flag),
            BlockDenseMHA(64, 128, 32,
                          mha_heads_num=4,
                          mha_range=self.mha_ranges[2],
                          back_time_flag=self.back_time_flag),
            BlockDenseMHA(128, 256, 64,
                          mha_heads_num=8,
                          mha_range=self.mha_ranges[3],
                          back_time_flag=self.back_time_flag),
        ])

        self.blocks_up = torch.nn.ModuleList([
            BlockDenseMHA(256, 128, 128,
                          mha_heads_num=4,
                          mha_range=self.mha_ranges[3],
                          back_time_flag=self.back_time_flag),
            BlockDenseMHA(128, 64, 64,
                          mha_heads_num=2,
                          mha_range=self.mha_ranges[2],
                          back_time_flag=self.back_time_flag),
            BlockDenseMHA(64, 32, 32,
                          mha_heads_num=1,
                          mha_range=self.mha_ranges[1],
                          back_time_flag=self.back_time_flag),
            BlockDenseMHA(32, 2, 16,
                          mha_heads_num=0,
                          mha_range=0,
                          back_time_flag=self.back_time_flag),
        ])

        ks = (3,3)
        pad_f = (ks[0]//2, ks[0]-1-ks[0]//2)
        if self.back_time_flag:
            pad_t = (ks[1] - 1, 0) # no future
        else:
            pad_t = (ks[1] // 2, ks[1] - 1 - ks[1] // 2)  # with future

        self.convs_up_pad = pad_t + pad_f

        self.convs_up = torch.nn.ModuleList([
            torch.nn.Conv2d(2*256, 256, kernel_size=ks),
            torch.nn.Conv2d(2*128, 128, kernel_size=ks),
            torch.nn.Conv2d(2*64,  64, kernel_size=ks),
            torch.nn.Conv2d(2*32,  32, kernel_size=ks),
        ])


        #create lowest level block
        inp_ch = self.blocks_down[-1].out_ch
        out_ch = self.blocks_up[0].inp_ch
        mha_heads_num = self.blocks_down[-1].mha_heads_num

        self.block_bottom = BlockDenseMHA(inp_ch, out_ch, inp_ch // 2,
                                          mha_heads_num,
                                          self.mha_ranges[4],
                                          back_time_flag=self.back_time_flag)

    def get_sample_ahead(self):
        return (self.ahead + self.ahead_ifft) * self.hop_length

    def get_sample_length_ceil(self, sample_len):
        t_step = 2 ** len(self.convs_up)
        ifp32 = sample_len / (t_step*self.hop_length)
        i = math.ceil(ifp32)
        return max(1, i) * t_step * self.hop_length

    def encode(self, x):
        # [B,T]->[B,1,1,T]
        x = x.unsqueeze(1).unsqueeze(2)
        # [B,1,1,T]->[B,2F,1,T]
        X = torch.nn.functional.conv2d(x, self._encdec_matrix, stride=self.hop_length)

        # [B,2F,1,T]->[B,2,F,T]
        B, F2, _, T = get_shape(X)
        return X.reshape(B,2,F2//2,T)

    def decode(self, X):
        B, _, F, T = get_shape(X)

        # [B,2,F,T] -> [B,2F,1,T]
        X = X.reshape(B, 2 * F, 1, T)

        # [B, 2F,1,T] -> [B,1,1,T]
        x = torch.nn.functional.conv_transpose2d(X, self._encdec_matrix, stride=self.hop_length)

        # [B, 1,1,T] -> [B,T]
        x = x[:,0,0,:]

        return x

    def forward(self, x, state_old=None):

        states = States(state_old)

        tail_size = self.wnd_length - self.hop_length
        x_padded = states.pad_left(x, tail_size, 1)

        X = self.encode(x_padded)
        # [B,2,F,T]
        z = X

        #DOWN
        skips = []
        for b in self.blocks_down:

            z,state = b(z, states.state_old)
            states.update(state)

            skips.append(z)
            z = self.pool(z)

        #BOTTOM
        z, state = self.block_bottom(z, states.state_old)
        states.update(state)

        #UP
        for skip, conv_up, block_up in zip(reversed(skips), self.convs_up, self.blocks_up):
            z = torch.nn.functional.interpolate(z, scale_factor=2, mode='nearest')
            Fs = get_shape(skip)[-2]
            Fz = get_shape(z)[-2]
            if Fz != Fs:
                z = torch.nn.functional.pad(z, (0,0,0,1), mode='replicate')
            z = torch.cat([z,skip],1)

            pad = self.convs_up_pad
            if pad[0]>0:
                z = states.pad_left(z, pad[0], 3)
                pad = (0,) + pad[1:]
            z = torch.nn.functional.pad(z, pad, mode='constant', value=0)
            z = conv_up(z)

            z,state = block_up(z, states.state_old)
            states.update(state)

        X = states.pad_left(X, self.ahead, 3, shift_right=True)

        # [B,2,F,T] -> [B,F,T],[B,F,T] ->
        Mr, Mi = z[:, 0], z[:, 1]
        Xr, Xi = X[:, 0], X[:, 1]

        # mask in complex space
        Yr = Xr * Mr - Xi * Mi
        Yi = Xr * Mi + Xi * Mr

        #[B,F,T] + [B,F,T] -> [B,2,F,T]
        Y = torch.stack([Yr,Yi], 1)

        # decode and return only valid samples
        Y_paded = states.pad_left(Y, self.ahead_ifft, 3)
        y = self.decode(Y_paded)
        y = y[:,tail_size:-self.ahead_ifft*self.hop_length]

        assert not states.state_old
        return y, Y, states.state
