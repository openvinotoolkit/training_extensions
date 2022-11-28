import torch
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics import PeakSignalNoiseRatio

def compare_psnr_batch(original, compressed, **kwargs):

    assert original.shape == compressed.shape, 'shapes should be same'
    assert len(original.shape) == 4  # Batch x Channel x Height x Width

    psnr = PeakSignalNoiseRatio()
    avg_psnr = psnr(compressed, original)

    return avg_psnr


def compare_ssim_batch(original, compressed, **kwargs):
    assert original.shape == compressed.shape # 'shapes should be same'
    assert len(original.shape) == 4  # Batch x Channel x Height x Width

    ssim = StructuralSimilarityIndexMeasure()
    avg_ssim = ssim(original, compressed)

    return avg_ssim
