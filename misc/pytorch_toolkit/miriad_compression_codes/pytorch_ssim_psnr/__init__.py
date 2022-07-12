import torch
import torch.nn.functional as F
from math import exp
import numpy as np
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics import PeakSignalNoiseRatio
from torch import Tensor


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(
        channel, 1, window_size, window_size).contiguous()
    return window


def reduce_tensor(x: Tensor, reduction='mean'):
    if reduction == 'mean':
        return x.mean()
    elif reduction == 'sum':
        return x.sum()
    return x


# Classes to re-use window

class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, reduction='mean', val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.reduction = reduction
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, input: Tensor, target: Tensor):
        (_, channel, _, _) = input.size()

        if channel == self.channel and self.window.dtype == input.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(
                input.device).type(input.dtype)
            self.window = window
            self.channel = channel
        ssim = StructuralSimilarityIndexMeasure()
        return ssim(preds=input, target=target, kernel=window, kernel_size=self.window_size, 
        reduction=self.reduction, data_rage=self.val_range)


class PSNR(torch.nn.Module):
    def __init__(self, reduction='mean', epsilon=1e-8, val_range=None):
        super(PSNR, self).__init__()
        self.reduction = reduction
        self.epsilon = epsilon
        self.val_range = val_range

    def forward(self, input: Tensor, target: Tensor):
        psnr = PeakSignalNoiseRatio()
        l = psnr(input, target, reduction=self.reduction)
        return reduce_tensor(l, self.reduction)
