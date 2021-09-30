import numpy as np
from nilearn.image import resample_to_img

def pad_image(image, mode='edge', pad_width=1):
    affine = np.copy(image.affine)
    spacing = np.copy(image.header.get_zooms()[:3])
    affine[:3, 3] -= spacing * pad_width
    if len(image.shape) > 3:
        # just pad the first three dimensions
        pad_width = [[pad_width]*2]*3 + [[0, 0]]*(len(image.shape) - 3)
    data = np.pad(image.get_data(), pad_width=pad_width, mode=mode)
    return image.__class__(data, affine)

def resample_image(source_image, target_image, interpolation="linear", pad_mode='edge', pad=False):
    if pad:
        source_image = pad_image(source_image, mode=pad_mode)
    return resample_to_img(source_image, target_image, interpolation=interpolation)

def resample(image, target_affine, target_shape, interpolation='linear', pad_mode='edge', pad=False):
    target_data = np.zeros(target_shape, dtype=image.get_data_dtype())
    target_image = image.__class__(target_data, affine=target_affine)
    return resample_image(image, target_image, interpolation=interpolation, pad_mode=pad_mode, pad=pad)
