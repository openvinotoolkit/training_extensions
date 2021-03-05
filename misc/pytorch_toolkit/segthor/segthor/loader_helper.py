# Copyright (C) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import os
import nibabel as nib
import numpy as np
from skimage import measure, morphology
import torch

# dataset statistics
mean = -303.0502877950004
mean2 = 289439.0029958802

def read_nii(filename):
    image = nib.load(filename)
    return np.array(image.get_data())

def read_numpy(filename):
    return np.load(filename)

def read_nii_header(filename):
    return nib.load(filename)


def get_indices(position, center_shape, border):
    index = [p * c for p, c in zip(position, center_shape)]
    index_min = [i - b for i, b in zip(index, border)]
    index_max = [i + c + b for i, c, b in zip(index, center_shape, border)]

    return index_min, index_max

def read_sample(path, filename, read_annotation=True):
    data_path = os.path.join(path, filename, filename + '.nii.gz')
    label_path = os.path.join(path, filename, 'GT.nii.gz')
    annotation = None

    header = read_nii_header(data_path)

    data = np.array(header.get_data()).astype(np.float32)
    if read_annotation:
        annotation = read_nii(label_path)

    return data, annotation, header


def lung_bbox(image):
    mean_image = image.max(axis=2) > 300
    mean_image = morphology.opening(mean_image, np.ones(shape=(5, 5)))
    projection = np.max(mean_image, axis=0)
    max_y = np.max(np.where(projection > 0))

    mask = (image > -400)
    mask = morphology.closing(mask, np.ones(shape=(10, 10, 10))) + 1
    label = measure.label(mask)
    background_label = label[0, 0, -1]
    mask[label == background_label] = 2
    mask = 2 - mask

    bbox = bbox3(mask)

    bbox[1, 2] = image.shape[2]
    bbox[0, 2] = 0
    bbox[1, 1] = max_y
    bbox[0, 1] = np.maximum(bbox[0, 1] - 10, 0)
    bbox[1, 0] = np.minimum(bbox[1, 0] + 10, image.shape[0])
    bbox[0, 0] = np.maximum(bbox[0, 0] - 10, 0)

    return bbox

def copy(data, tile_shape, index_min, index_max):
    ret = torch.zeros(size=data.shape[:2]+tuple(tile_shape), dtype=torch.float)

    index_clamp_min = np.maximum(index_min, 0)
    index_clamp_max = np.minimum(index_max, data.shape[2:])

    diff_min = [min_c - min_i for min_c, min_i in zip(index_clamp_min, index_min)]
    diff_max = [t - (max_i - max_c) for t, max_c, max_i in zip(tile_shape, index_clamp_max, index_max)]

    ret[:, :, diff_min[0]:diff_max[0], diff_min[1]:diff_max[1], diff_min[2]:diff_max[2]] = \
        data[:, :, index_clamp_min[0]:index_clamp_max[0], index_clamp_min[1]:index_clamp_max[1],
             index_clamp_min[2]:index_clamp_max[2]]

    return ret


def ravel_index(index, grid):
    i = 0
    prod = 1
    for j in reversed(range(len(grid))):
        i = i + prod * index[j]
        prod = prod * grid[j]

    return i


def unravel_index(index, grid):
    i = []
    prod = np.prod(grid)
    for gr in grid:
        prod = prod // gr
        i.append(index // prod)
        index = index % prod
    return i


def copy_back(data, tile, center_shape, index_min, index_max, border):
    index_center_min = [i + b for i, b in zip(index_min, border)]
    index_center_max = [i - b for i, b in zip(index_max, border)]

    index_clamp_min = np.maximum(index_center_min, 0)
    index_clamp_max = np.minimum(index_center_max, data.shape[2:])

    diff_min = [t + min_c - min_i for t, min_c, min_i in zip(border, index_clamp_min, index_center_min)]
    diff_max = [b + t - (max_i - max_c) for b, t, max_c, max_i in
                zip(border, center_shape, index_clamp_max, index_center_max)]

    data[:, :, index_clamp_min[0]:index_clamp_max[0], index_clamp_min[1]:index_clamp_max[1],
         index_clamp_min[2]:index_clamp_max[2]] = \
        tile[:, :, diff_min[0]:diff_max[0], diff_min[1]:diff_max[1], diff_min[2]:diff_max[2]]


def closest_to_k(n, k=8):
    if n % k == 0:
        return n
    return ((n // k) + 1)*k


def leave_biggest_region(connectivity):
    unique, counts = np.unique(connectivity, return_counts=True)
    counts_idx = np.argsort(-counts)
    unique = unique[counts_idx]

    resulting_conectivity = np.zeros_like(connectivity)

    resulting_conectivity[connectivity == unique[1]] = 1

    return resulting_conectivity


def reject_small_regions(connectivity, ratio=0.25):
    resulting_connectivity = connectivity.copy()
    unique, counts = np.unique(connectivity, return_counts=True)

    all_nonzero_clusters = np.prod(connectivity.shape) - np.max(counts)

    for i in range(unique.shape[0]):
        if counts[i] < ratio * all_nonzero_clusters:
            resulting_connectivity[resulting_connectivity == unique[i]] = 0

    return resulting_connectivity

def bbox3(img):
    """
    compute bounding box of the nonzero image pixels
    :param img: input image
    :return: bbox with shape (2, 3) and contents [min, max]
    """
    rows = np.any(img, axis=1)
    rows = np.any(rows, axis=1)

    cols = np.any(img, axis=0)
    cols = np.any(cols, axis=1)

    slices = np.any(img, axis=0)
    slices = np.any(slices, axis=0)

    rows = np.where(rows)
    cols = np.where(cols)
    slices = np.where(slices)
    if rows[0].shape[0] > 0:
        rmin, rmax = rows[0][[0, -1]]
        cmin, cmax = cols[0][[0, -1]]
        smin, smax = slices[0][[0, -1]]

        return np.array([[rmin, cmin, smin], [rmax, cmax, smax]])
    return np.array([[-1, -1, -1], [0, 0, 0]])
