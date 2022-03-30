# File from Glow-TTS https://github.com/jaywalnut310/glow-tts
# Copyright (c) 2020 Jaehyeon Kim
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2017, Prem Seetharaman
# SPDX-License-Identifier: BSD-3-Clause
#
import torch
import numpy as np
import torch.nn.functional as F
from scipy.signal import get_window
from librosa.util import pad_center, tiny
from librosa import stft, istft

from .audio_processing import window_sumsquare


class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
    def __init__(self, filter_length=800, hop_length=200, win_length=800,
                 window='hann'):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        if window is not None:
            assert(filter_length >= win_length)
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        if input_data.device.type == "cuda":
            # similar to librosa, reflect-pad the input
            input_data = input_data.view(num_batches, 1, num_samples)
            input_data = F.pad(
                input_data.unsqueeze(1),
                (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
                mode='reflect')
            input_data = input_data.squeeze(1)

            forward_transform = F.conv1d(
                input_data,
                self.forward_basis,
                stride=self.hop_length,
                padding=0)

            cutoff = int((self.filter_length / 2) + 1)
            real_part = forward_transform[:, :cutoff, :]
            imag_part = forward_transform[:, cutoff:, :]
        else:
            x = input_data.detach().numpy()
            real_part = []
            imag_part = []
            for y in x:
                y_ = stft(y, self.filter_length, self.hop_length, self.win_length, self.window)
                real_part.append(y_.real[None,:,:])
                imag_part.append(y_.imag[None,:,:])
            real_part = np.concatenate(real_part, 0)
            imag_part = np.concatenate(imag_part, 0)

            real_part = torch.from_numpy(real_part).to(input_data.dtype)
            imag_part = torch.from_numpy(imag_part).to(input_data.dtype)

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.atan2(imag_part.data, real_part.data)

        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)

        if magnitude.device.type == "cuda":
            inverse_transform = F.conv_transpose1d(
                recombine_magnitude_phase,
                self.inverse_basis,
                stride=self.hop_length,
                padding=0)

            if self.window is not None:
                window_sum = window_sumsquare(
                    self.window, magnitude.size(-1), hop_length=self.hop_length,
                    win_length=self.win_length, n_fft=self.filter_length,
                    dtype=np.float32)
                # remove modulation effects
                approx_nonzero_indices = torch.from_numpy(
                    np.where(window_sum > tiny(window_sum))[0])
                window_sum = torch.from_numpy(window_sum).to(inverse_transform.device)
                inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

                # scale by hop ratio
                inverse_transform *= float(self.filter_length) / self.hop_length

            inverse_transform = inverse_transform[:, :, int(self.filter_length/2):]
            inverse_transform = inverse_transform[:, :, :-int(self.filter_length/2):]
            inverse_transform = inverse_transform.squeeze(1)
        else:
            x_org = recombine_magnitude_phase.detach().numpy()
            n_b, n_f, n_t = x_org.shape
            x = np.empty([n_b, n_f//2, n_t], dtype=np.complex64)
            x.real = x_org[:,:n_f//2]
            x.imag = x_org[:,n_f//2:]
            inverse_transform = []
            for y in x:
                y_ = istft(y, self.hop_length, self.win_length, self.window)
                inverse_transform.append(y_[None,:])
            inverse_transform = np.concatenate(inverse_transform, 0)
            inverse_transform = torch.from_numpy(inverse_transform).to(recombine_magnitude_phase.dtype)

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction
