# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

import cython
import numpy as np

cimport numpy as np

np.import_array()
import cv2
from PIL import Image

ctypedef np.int32_t INT32_t
ctypedef np.uint8_t UINT8_t


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[INT32_t, ndim=1] c_histogram(const UINT8_t[:, :, :] image):
    cdef np.ndarray[INT32_t, ndim=1] hist = np.zeros((768,), dtype=np.int32)
    cdef INT32_t [:] hist_view = hist
    cdef int height, width, y, x
    cdef UINT8_t r, g, b

    height = image.shape[0]
    width = image.shape[1]

    for y in range(height):
        for x in range(width):
            hist_view[image[y][x][0] + 000] += 1
            hist_view[image[y][x][1] + 256] += 1
            hist_view[image[y][x][2] + 512] += 1

    return hist


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _c_lut(np.ndarray[UINT8_t, ndim=3] image, const UINT8_t[:] r_lut):
    assert np.PyArray_ISCONTIGUOUS(image), "Image should be contiguous."
    assert image.ndim == 3, "Image should be 3D array."
    assert image.shape[2] == 3, "Image should have 3 channels."

    cdef int h = image.shape[0]
    cdef int w = image.shape[1]
    cdef int x, y, curr_idx
    cdef int r, g, b
    cdef UINT8_t* data_ptr = &image[0,0,0]

    for y in range(h):
        for x in range(w):
            r = image[y, x, 0] + 000
            g = image[y, x, 1] + 256
            b = image[y, x, 2] + 512

            curr_idx = 3 * w * y + 3 * x

            data_ptr[curr_idx + 0] = r_lut[r]
            data_ptr[curr_idx + 1] = r_lut[g]
            data_ptr[curr_idx + 2] = r_lut[b]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def autocontrast(np.ndarray[UINT8_t, ndim=3] image, cutoff=0, ignore=None) -> np.ndarray:
    cdef int* h
    cdef int i, lo, hi, ix, cut, layer
    cdef double scale, offset
    cdef int[:] histogram
    cdef np.ndarray[UINT8_t, ndim=1] lut = np.zeros((768,), dtype=np.uint8)
    cdef UINT8_t[:] lut_view = lut
    cdef np.ndarray[UINT8_t, ndim=3] result

    histogram = c_histogram(image)

    for layer in range(0, 768, 256):
        h = &histogram[layer]

        if ignore is not None:
            # get rid of outliers
            try:
                h[ignore] = 0
            except TypeError:
                # assume sequence
                for ix in ignore:
                    h[ix] = 0
        if cutoff:
            # cut off pixels from both ends of the histogram
            if not isinstance(cutoff, tuple):
                cutoff = (cutoff, cutoff)
            # get number of pixels
            n = 0
            for ix in range(256):
                n = n + h[ix]
            # remove cutoff% pixels from the low end
            cut = n * cutoff[0] // 100
            for lo in range(256):
                if cut > h[lo]:
                    cut = cut - h[lo]
                    h[lo] = 0
                else:
                    h[lo] -= cut
                    cut = 0
                if cut <= 0:
                    break
            # remove cutoff% samples from the high end
            cut = n * cutoff[1] // 100
            for hi in range(255, -1, -1):
                if cut > h[hi]:
                    cut = cut - h[hi]
                    h[hi] = 0
                else:
                    h[hi] -= cut
                    cut = 0
                if cut <= 0:
                    break
        # find lowest/highest samples after preprocessing
        for lo in range(256):
            if h[lo]:
                break
        for hi in range(255, -1, -1):
            if h[hi]:
                break
        if hi <= lo:
            # don't bother
            for i in range(256):
                lut_view[layer + i] = i
        else:
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            for ix in range(256):
                i = ix
                ix = (int)(ix * scale + offset)
                if ix < 0:
                    ix = 0
                elif ix > 255:
                    ix = 255
                lut_view[layer + i] = ix

        layer += 256

    _c_lut(image, lut_view)
    return image
