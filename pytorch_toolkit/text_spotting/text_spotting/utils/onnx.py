"""
 Copyright (c) 2019 Intel Corporation

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

import numpy as np
import onnx

import torch
import torch.nn as nn


def get_input_size(image_size, scale, max_size, divisor=1):
    image_size = np.asarray(image_size)

    # Get scale factor for image resize.
    im_scale = float(scale) / float(min(image_size))
    if np.round(im_scale * max(image_size)) > max_size:
        im_scale = float(max_size) / float(max(image_size))

    # Get resized image size.
    resized_image_size = image_size * im_scale
    input_size = np.round(resized_image_size).astype(np.int)

    # Pad the image for size be a multiple of `divisor`.
    input_size[0] = (input_size[0] + divisor - 1) // divisor * divisor
    input_size[1] = (input_size[1] + divisor - 1) // divisor * divisor

    return input_size


def onnx_export(net, input_size, output_folder, check=False, verbose=False):
    os.makedirs(output_folder, exist_ok=True)

    net.eval()
    if hasattr(net, 'force_max_output_size'):
        # Force model to output maximal possible number of proposals and detections.
        net.force_max_output_size = True
    # Use ONNX stubs for all the blocks that don't support direct export.
    for m in net.modules():
        if hasattr(m, 'use_stub'):
            m.use_stub = True

    im_data = np.random.randn(1, 3, input_size[0], input_size[1]).astype(np.float32)
    im_info = np.asarray([[input_size[0], input_size[1], 1.0]], dtype=np.float32)
    im_info = torch.tensor(im_info)
    im_data = torch.tensor(im_data)
    if torch.cuda.is_available():
        net = net.cuda()
        im_info = im_info.cuda()
        im_data = im_data.cuda()
    im_data.requires_grad = True
    im_info.requires_grad = True

    input_names = ['im_data', 'im_info']
    outputs_names = ['boxes', 'classes', 'scores', 'batch_ids', 'raw_masks', 'text_features']
    torch.onnx.export(net, (im_data, im_info), os.path.join(output_folder, 'detector.onnx'),
                      verbose=verbose, input_names=input_names, output_names=outputs_names)

    net_from_onnx = onnx.load(os.path.join(output_folder, 'detector.onnx'))
    if check:
        try:
            onnx.checker.check_model(net_from_onnx)
            logging.info('ONNX check passed.')
        except onnx.onnx_cpp2py_export.checker.ValidationError as ex:
            logging.warning('ONNX check failed.')
            logging.warning(ex)

    return onnx.helper.printable_graph(net_from_onnx.graph)


def export_to_onnx_text_recognition_encoder(net, input_size, output_folder, check=False):
    os.makedirs(output_folder, exist_ok=True)

    net.eval()
    dim = net.dim_input
    input = np.random.randn(1, dim, *input_size).astype(np.float32)
    input = torch.tensor(input)
    if torch.cuda.is_available():
        net = net.cuda()
        input = input.cuda()
    input.requires_grad = True
    torch.onnx.export(net, input, os.path.join(output_folder, 'encoder.onnx'), verbose=True,
                      input_names=['input'], output_names=['output'])

    net_from_onnx = onnx.load(os.path.join(output_folder, 'encoder.onnx'))
    if check:
        try:
            onnx.checker.check_model(net_from_onnx)
            logging.info('ONNX check passed.')
        except onnx.onnx_cpp2py_export.checker.ValidationError as ex:
            logging.warning('ONNX check failed.')
            logging.warning(ex)

    return onnx.helper.printable_graph(net_from_onnx.graph)


def export_to_onnx_text_recognition_decoder(net, input_size, output_folder, check=False):
    os.makedirs(output_folder, exist_ok=True)

    net.eval()
    dim = net.hidden_size
    prev_input = np.random.randn(1).astype(np.float32)
    pev_hidden = np.random.randn(1, 1, dim).astype(np.float32)
    prev_cell = np.random.randn(1, 1, dim).astype(np.float32)
    encoder_outputs = np.random.randn(1, input_size[0] * input_size[1], dim).astype(np.float32)
    prev_input = torch.tensor(prev_input)
    pev_hidden = torch.tensor(pev_hidden)
    prev_cell = torch.tensor(prev_cell)
    encoder_outputs = torch.tensor(encoder_outputs)
    if torch.cuda.is_available():
        net = net.cuda()
        prev_input = prev_input.cuda()
        pev_hidden = pev_hidden.cuda()
        prev_cell = prev_cell.cuda()
        encoder_outputs = encoder_outputs.cuda()
    prev_input.requires_grad = False
    pev_hidden.requires_grad = True
    prev_cell.requires_grad = True
    encoder_outputs.requires_grad = True
    if isinstance(net.decoder, nn.GRU):
        torch.onnx.export(net, (prev_input, pev_hidden, encoder_outputs),
                          os.path.join(output_folder, 'decoder.onnx'), verbose=True,
                          input_names=['prev_symbol', 'prev_hidden', 'encoder_outputs'],
                          output_names=['output', 'hidden', 'attention']
                          )
    elif isinstance(net.decoder, nn.LSTM):
        torch.onnx.export(net, (prev_input, pev_hidden, encoder_outputs, prev_cell),
                          os.path.join(output_folder, 'decoder.onnx'), verbose=True,
                          input_names=['prev_symbol', 'prev_hidden', 'encoder_outputs',
                                       'prev_cell'],
                          output_names=['output', 'hidden', 'cell', 'attention']
                          )

    net_from_onnx = onnx.load(os.path.join(output_folder, 'decoder.onnx'))
    if check:
        try:
            onnx.checker.check_model(net_from_onnx)
            logging.info('ONNX check passed.')
        except onnx.onnx_cpp2py_export.checker.ValidationError as ex:
            logging.warning('ONNX check failed.')
            logging.warning(ex)

    return onnx.helper.printable_graph(net_from_onnx.graph)
