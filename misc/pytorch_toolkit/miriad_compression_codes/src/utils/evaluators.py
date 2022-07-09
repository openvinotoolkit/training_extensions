import torch
# from skimage.measure import compare_psnr  # no longer works version 0.16 onwards
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import pytorch_ssim_psnr


def compare_psnr_batch(original, compressed, **kwargs):

    assert original.shape == compressed.shape, 'shapes should be same'
    assert len(original.shape) == 4  # Batch x Channel x Height x Width

    avg_psnr = 0.0
    batch_size, n = original.shape[0], 0

    for idx in range(original.shape[0]):
        # take each image in the bacth one by one

        # minor shape adjustment for being compatible with 'compare_psnr()' function
        one_original = original[idx, 0, ...]
        one_compressed = compressed[idx, 0, ...]

        # measure pSNR
        psnr = compare_psnr(one_original, one_compressed, **kwargs)

        # running average on the individual pSNRs
        avg_psnr = ((n * avg_psnr) + psnr) / (n + 1)
        n += 1

    return avg_psnr


def compare_ssim_batch(original, compressed, **kwargs):

    assert original.shape == compressed.shape, 'shapes should be same'
    assert len(original.shape) == 4  # Batch x Channel x Height x Width

    avg_ssim = pytorch_ssim_psnr.msssim(
        torch.tensor(original, requires_grad=False),
        torch.tensor(compressed, requires_grad=False), **kwargs).item()

    return avg_ssim
