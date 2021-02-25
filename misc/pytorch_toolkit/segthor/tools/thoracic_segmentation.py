#!/usr/bin/env python3
#
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
import logging
from sys import stdout

from datetime import datetime
from argparse import ArgumentParser, SUPPRESS
from scipy.ndimage import zoom
from scipy.ndimage.filters import median_filter
from skimage import measure, morphology

import numpy as np
import nibabel as nii
from openvino.inference_engine import IENetwork, IEPlugin

logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.DEBUG, stream=stdout)
logger = logging.getLogger('thoracic_segmentation_demo')


def parse_arguments():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-i', '--path_to_input_data', type=str, required=True,
                      help="Required. Path to an input folder with NIfTI data/TIFF file")
    args.add_argument('-m', '--path_to_model', type=str, required=True,
                      help="Required. Path to an .xml file with a trained model")
    args.add_argument('-o', '--path_to_output', type=str, required=True,
                      help="Required. Path to a folder where output files will be saved")
    args.add_argument('-d', '--target_device', type=str, required=False, default="CPU",
                      help="Optional. Specify a target device to infer on: CPU. "
                           "Use \"-d HETERO:<comma separated devices list>\" format to specify HETERO plugin.")
    args.add_argument('-l', '--path_to_extension', type=str, required=False, default=None,
                      help="Required for CPU custom layers. "
                           "Absolute path to a shared library with the kernels implementations.")
    args.add_argument('-nthreads', '--number_threads', type=int, required=False, default=None,
                      help="Optional. Number of threads to use for inference on CPU (including HETERO cases).")
    args.add_argument('-c', '--path_to_cldnn_config', type=str, required=False,
                      help="Required for GPU custom kernels. "
                           "Absolute path to an .xml file with the kernels description.")
    return parser.parse_args()


def read_nii_header(filename):
    return nii.load(filename)


def bbox3(img):
    """
    compute bounding box of the nonzero image pixels
    :param img: input image
    :return: bbox with shape (2,3) and contents [min,max]
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


# pylint: disable=R0914,R0915
def main():
    args = parse_arguments()

    # --------------------------------- 1. Load Plugin for inference engine ---------------------------------
    logger.info("Loading plugin")
    plugin = IEPlugin(args.target_device)

    config = dict()
    if 'CPU' in args.target_device:
        if args.path_to_extension:
            plugin.add_cpu_extension(args.path_to_extension)
        if args.number_threads is not None:
            config.update({'CPU_THREADS_NUM': str(args.number_threads)})
    else:
        raise AttributeError("Device {} do not support of 3D convolution. Please use CPU or HETERO:*CPU*")

    if 'GPU' in args.target_device:
        if args.path_to_cldnn_config:
            config.update({'CONFIG_FILE': args.path_to_cldnn_config})
            logger.info("GPU extensions is loaded %s", args.path_to_cldnn_config)

    plugin.set_config(config)

    logger.info("Device is %s ", plugin.device)
    logger.info("Plugin version is %s", plugin.version)

    # --------------------- 2. Read IR Generated by ModelOptimizer (.xml and .bin files) ---------------------

    xml_filename = os.path.abspath(args.path_to_model)
    bin_filename = os.path.abspath(os.path.splitext(xml_filename)[0] + '.bin')

    ie_network = IENetwork(xml_filename, bin_filename)

    input_info = ie_network.inputs
    if not input_info:
        raise AttributeError("No inputs info is provided")
    if len(input_info) != 1:
        raise AttributeError("Only one input layer network is supported")

    input_name = next(iter(input_info))
    out_name = next(iter(ie_network.outputs))
    print(input_name, out_name)

    # ---------------------------------------- 4. Preparing input data ----------------------------------------
    logger.info("Preparing inputs")

    if len(input_info[input_name].shape) != 5:
        raise AttributeError("Incorrect shape {} for 3d convolution network".format(args.shape))

    n, _, d, h, w = input_info[input_name].shape
    ie_network.batch_size = n

    # ------------------------------------- 4. Loading model to the plugin -------------------------------------
    # logger.info("Reshape of network from {} to {}".format(input_info[input_name].shape, image_crop_pad.shape))
    #ie_network.reshape({input_name: image_crop_pad.shape})
    #input_info = ie_network.inputs

    # logger.info("Loading model to the plugin")
    executable_network = plugin.load(network=ie_network)
    del ie_network

    files = os.listdir(args.path_to_input_data)
    files = [f for f in files if (f.startswith('Patient') and os.path.isfile(os.path.join(args.path_to_input_data, f)))]
    files.sort()

    for f in files:
        header = read_nii_header(os.path.join(args.path_to_input_data, f))
        image = np.array(header.get_data()).astype(np.float32)
        original_shape = image.shape

        start_time = datetime.now()

        image = median_filter(image, 3)

        bbox = lung_bbox(image)

        image_crop = image[bbox[0, 0]:bbox[1, 0], bbox[0, 1]:bbox[1, 1], bbox[0, 2]:bbox[1, 2]]

        new_shape_pad = (d, h, w)
        diff = np.array(new_shape_pad) - np.array(image_crop.shape)
        pad_left = diff // 2
        pad_right = diff - pad_left

        image_crop_pad = np.pad(image_crop, pad_width=tuple([(pad_left[i], pad_right[i]) for i in range(3)]),
                                mode='reflect')

        # dataset statistics
        mean = -303.0502877950004
        mean2 = 289439.0029958802
        std = np.sqrt(mean2 - mean * mean)

        image_crop_pad = (image_crop_pad - mean) / std

        image_crop_pad = image_crop_pad[None, None]

        preprocess_time = datetime.now() - start_time

        test_im = {input_name: image_crop_pad}

        # ---------------------------------------------- 5. Do inference --------------------------------------------
        start_time = datetime.now()
        res = executable_network.infer(test_im)
        infer_time = datetime.now() - start_time

        # ---------------------------- 6. Processing of the received inference results ------------------------------
        result = res[out_name]

        start_time = datetime.now()

        output_crop = result[0, :, pad_left[0]:-pad_right[0] or None, pad_left[1]:-pad_right[1] or None,
                             pad_left[2]:-pad_right[2] or None]

        new_label = np.zeros(shape=(4,) + image.shape)
        new_label[:, bbox[0, 0]:bbox[1, 0], bbox[0, 1]:bbox[1, 1], bbox[0, 2]:bbox[1, 2]] = output_crop

        scale_factor = np.array(original_shape) / np.array(image.shape)
        old_labels = [zoom(new_label[i], scale_factor, order=1, mode='constant', cval=0)[None] for i in range(4)]

        old_label = np.concatenate(tuple(old_labels), axis=0)
        old_label = ((np.argmax(old_label, axis=0) + 1) *
                     np.max((old_label > np.array([0.5, 0.5, 0.5, 0.5]).reshape((-1, 1, 1, 1))).astype(np.int32),
                            axis=0)).astype(np.int32)

        eso_connectivity = morphology.label(old_label == 1)
        heart_connectivity = morphology.label(old_label == 2)
        trachea_connectivity = morphology.label(old_label == 3)
        aorta_connectivity = morphology.label(old_label == 4)
        eso_connectivity = reject_small_regions(eso_connectivity, ratio=0.2)
        heart_connectivity = leave_biggest_region(heart_connectivity)
        trachea_connectivity = leave_biggest_region(trachea_connectivity)
        aorta_connectivity = leave_biggest_region(aorta_connectivity)

        old_label[np.logical_and(old_label == 1, eso_connectivity == 0)] = 0
        old_label[np.logical_and(old_label == 2, heart_connectivity == 0)] = 0
        old_label[np.logical_and(old_label == 3, trachea_connectivity == 0)] = 0
        old_label[np.logical_and(old_label == 4, aorta_connectivity == 0)] = 0

        postprocess_time = datetime.now() - start_time

        logger.info("Pre-processing time is %s; Inference time is %s; Post-processing time is %s",
                    preprocess_time, infer_time, postprocess_time)

        # --------------------------------------------- 7. Save output -----------------------------------------------
        output_header = nii.Nifti1Image(old_label, header.affine)
        nii.save(output_header, os.path.join(args.path_to_output, f[:-7]+'.nii'))

if __name__ == "__main__":
    main()
