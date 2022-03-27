# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# torchaudio:
# Copyright (c) 2017 Facebook Inc. (Soumith Chintala)
# SPDX-License-Identifier: BSD-2-Clause
#
# This file contains parts based on torchaudio/transforms.py from https://github.com/pytorch/audio v0.10.1

import enum
import random
from typing import Callable, Optional, Tuple

import scipy
import librosa
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torchaudio
from torchaudio import functional as F
from torchaudio.transforms import Spectrogram


######################
## Waveform transforms
######################

class WaveformTransformSOX:
    def __init__(self, p):
        self.p = p

    def __call__(self, waveform, sample_rate):
        if random.random() < self.p:
            waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate, self.get_effect()
            )
        return waveform, sample_rate

    def uniform(self, val_min, val_max):
        return val_min + (val_max - val_min) * random.random()

    def randint(self, val_min, val_max):
        return random.randint(val_min, val_max)


class WaveformTransformSpeed(WaveformTransformSOX):
    def __init__(self, p=1.0, val_min=0.7, val_max=2.0):
        super().__init__(p)
        self.val_min = val_min
        self.val_max = val_max

    def get_effect(self):
        return [["speed", str(self.uniform(self.val_min, self.val_max))]]


class WaveformTransformResample(WaveformTransformSOX):
    def __init__(self, p=1.0, rates=[8000, 16000, 32000]):
        super().__init__(p)
        self.rates = rates

    def get_effect(self):
        return [["rate", str(random.choice(self.rates))]]


class WaveformTransformGain(WaveformTransformSOX):
    def __init__(self, p=1.0, val_min=1.0, val_max=10.0):
        super().__init__(p)
        self.val_min = val_min
        self.val_max = val_max

    def get_effect(self):
        return [["gain", str(self.uniform(self.val_min, self.val_max))]]


class WaveformTransformPitch(WaveformTransformSOX):
    def __init__(self, p=1.0, val_min=-1000.0, val_max=1000.0):
        super().__init__(p)
        self.val_min = val_min
        self.val_max = val_max

    def get_effect(self):
        return [["pitch", str(self.uniform(self.val_min, self.val_max))]]


class WaveformTransformLowpass(WaveformTransformSOX):
    def __init__(self, p=1.0, val_min=0, val_max=1000):
        super().__init__(p)
        self.val_min = val_min
        self.val_max = val_max

    def get_effect(self):
        return [["lowpass", "-1", str(self.randint(self.val_min, self.val_max))]]


#########################
## Spectrogram transforms
#########################

def normalize_uniform(x):
    return (x - torch.mean(x, dim=1, keepdim=True)) / (torch.std(x, dim=1, keepdim=True) + 1e-18)


# This class contains modified code from torchaudio/transforms.py from https://github.com/pytorch/audio v0.10.1
class MelSpectrogramV1(nn.Module):
    def __init__(self,
                 n_fft: int = 400,
                 win_length: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 f_min: float = 0.,
                 f_max: Optional[float] = None,
                 pad: int = 0,
                 n_mels: int = 128,
                 window_fn: Callable[..., Tensor] = torch.hann_window,
                 power: float = 2.,
                 normalized: bool = False,
                 wkwargs: Optional[dict] = None,
                 center: bool = True,
                 pad_mode: str = "reflect",
                 onesided: bool = True,
                 norm: Optional[str] = None,
                 mel_scale: str = "htk",
                 mel_norm: str = "none"
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.n_stft = self.n_fft // 2 + 1
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.pad = pad
        self.power = power
        self.normalized = normalized
        self.n_mels = n_mels  # number of mel frequency bins
        self.f_max = f_max
        self.f_min = f_min
        self.spectrogram = Spectrogram(n_fft=self.n_fft, win_length=self.win_length,
                                       hop_length=self.hop_length,
                                       pad=self.pad, window_fn=window_fn, power=self.power,
                                       normalized=self.normalized, wkwargs=wkwargs,
                                       center=center, pad_mode=pad_mode, onesided=onesided)
        self.mel_scale = mel_scale
        self.norm = norm
        self.fb = {}
        if mel_norm == "uniform":
            self.mel_normalize = normalize_uniform
        else:
            self.mel_normalize = None

    @torch.no_grad()
    def forward(self, waveform, sample_rate):
        spec = self.spectrogram(waveform)
        mel_spec = self._mel_scale(spec, sample_rate)
        if self.mel_normalize is not None:
            mel_spec = self.mel_normalize(torch.log(torch.clamp(mel_spec, min=1e-18)))
        return mel_spec, sample_rate

    def _mel_scale(self, spec, sample_rate):
        # pack batch
        shape = spec.size()
        spec = spec.reshape(-1, shape[-2], shape[-1])
        # (channel, frequency, time).transpose(...) dot (frequency, n_mels)
        # -> (channel, time, n_mels).transpose(...)
        mel_spec = torch.matmul(spec.transpose(1, 2), self._get_fb(spec, sample_rate)).transpose(1, 2)
        # unpack batch
        mel_spec = mel_spec.reshape(shape[:-2] + mel_spec.shape[-2:])
        return mel_spec

    def _get_fb(self, spec, sample_rate):
        if not f"fb_{sample_rate}" in self.fb:
            f_min = self.f_min
            f_max = self.f_max if self.f_max is not None else float(sample_rate // 2)
            assert f_min <= f_max, 'Require f_min: {} < f_max: {}'.format(f_min, f_max)
            fb = F.create_fb_matrix(
                spec.size(1), f_min, f_max,
                self.n_mels, sample_rate, self.norm,
                self.mel_scale
            ).to(spec.device)
            self.fb[f"fb_{sample_rate}"] = fb
        if self.fb[f"fb_{sample_rate}"].device != spec.device:
            self.fb[f"fb_{sample_rate}"] = self.fb[f"fb_{sample_rate}"].to(spec.device)
        return self.fb[f"fb_{sample_rate}"]


class MelSpectrogram:
    def __init__(
            self,
            preemph = 0.97,
            n_fft = 512,
            win_length = 0.02,
            hop_length = 0.01,
            center = True,
            pad_mode = 'reflect',
            n_mels = 64,
            fmin = 0.0,
            fmax = 8000.0,
            norm = 'slaney',
            htk = False
    ):
            self.preemph = preemph
            self.n_fft = n_fft
            self.win_length = win_length
            self.hop_length = hop_length
            self.center = center
            self.pad_mode = pad_mode
            self.n_mels = n_mels
            self.fmin = fmin
            self.fmax = fmax
            self.norm = norm
            self.htk = htk

    def __call__(self, audio, sample_rate=16000, expand_dims=True):
        assert sample_rate == 16000, f"Error: sample_rate = {sample_rate}. Only 16 KHz audio supported"
        preemphased = np.concatenate([audio[:1], audio[1:] - self.preemph * audio[:-1].astype(np.float32)])
        win_length = round(sample_rate * self.win_length)
        spec = np.abs(librosa.core.spectrum.stft(
            preemphased,
            n_fft=self.n_fft,
            hop_length=round(sample_rate * self.hop_length),
            win_length=win_length,
            center=self.center,
            window=scipy.signal.windows.hann(win_length),
            pad_mode=self.pad_mode
        ))
        mel_basis = librosa.filters.mel(
            sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            norm=self.norm,
            htk=self.htk
        )
        log_melspectrum = np.log(np.dot(mel_basis, np.power(spec, 2)) + 2 ** -24)
        normalized = (log_melspectrum - log_melspectrum.mean(1)[:, None]) / (log_melspectrum.std(1)[:, None] + 1e-5)
        if expand_dims:
            normalized = np.expand_dims(normalized, 0)
        return normalized, sample_rate



class MelMaskTransform(nn.Module):
    def __init__(
            self,
            freq_min: int = 0,
            freq_max: int = 0,
            time_min:int = 0,
            time_max: int = 0,
            fill_value: int = 0,
            p: float = 1.0
    ):
        super().__init__()
        self.p = p
        self.freq_min, self.freq_max = freq_min, freq_max
        self.time_min, self.time_max = time_min, time_max
        self.fill_value=fill_value

    def generate_line(self, shape: torch.Size, val_min: int, val_max: int) -> Tuple[int, int]:
        width = random.randint(val_min, val_max + 1)
        pos = int(random.uniform(0, shape - width))
        return pos, width

    def generate_line_freq(self, shape: torch.Size) -> Tuple[int, int]:
        return self.generate_line(shape, self.freq_min, self.freq_max)

    def generate_line_time(self, shape: torch.Size) -> Tuple[int, int]:
        return self.generate_line(shape, self.time_min, self.time_max)

    def need_apply(self) -> bool:
        if random.random() < self.p:
            return True
        else:
            return False


class MelSpecTransform(MelMaskTransform):
    def __init__(self, freq_holes=10, time_holes=10, freq_min=0, freq_max=0, time_min=0, time_max=0, fill_value=0, p=1.0):
        super().__init__(freq_min, freq_max, time_min, time_max, fill_value)
        self.freq_holes = freq_holes
        self.time_holes = time_holes

    @torch.no_grad()
    def forward(self, x, sample_rate):
        if not self.need_apply():
            return x
        mask = torch.zeros(x.size(), dtype=torch.bool, device=x.device)
        for idx in range(x.size(0)):
            holes = random.randint(0, self.freq_holes)
            for i in range(holes):
                f0, w = self.generate_line_freq(x.size(1))
                mask[idx, f0:f0+w] = 1

            holes = random.randint(0, self.time_holes)
            for i in range(holes):
                t0, h = self.generate_line_time(x.size(2))
                mask[idx, :, t0:t0+h] = 1

        return x.masked_fill(mask, self.fill_value), sample_rate


class MelCutoutTransform(MelMaskTransform):
    def __init__(self, holes=0, freq_min=0, freq_max=0, time_min=0, time_max=0, fill_value=0, p=1.0):
        super().__init__(freq_min, freq_max, time_min, time_max, fill_value)
        self.holes = holes

    @torch.no_grad()
    def forward(self, x, sample_rate):
        if not self.need_apply():
            return x
        mask = torch.zeros(x.size(), dtype=torch.bool, device=x.device)
        for idx in range(x.size(0)):
            holes = random.randint(0, self.holes)
            for i in range(holes):
                f0, w = self.generate_line_freq(x.size(1))
                t0, h = self.generate_line_time(x.size(2))
                mask[idx, f0:f0+w, t0:t0+h] = 1

        return x.masked_fill(mask, self.fill_value), sample_rate
