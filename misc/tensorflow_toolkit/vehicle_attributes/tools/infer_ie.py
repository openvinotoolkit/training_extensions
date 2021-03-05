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

from __future__ import print_function
import sys
import os
from argparse import ArgumentParser
import logging as log
import numpy as np
import cv2

from openvino.inference_engine import IENetwork, IEPlugin

CLASS_MAP = {
  0: 'car',
  1: 'bus',
  2: 'track',
  3: 'van',
}

def normalized_to_absolute(prediction):
  colorcar = np.zeros((1, 1, 3), dtype=np.uint8)
  for i in range(3):
    if prediction[i] < 0:
      colorcar[0, 0, i] = 0
    elif prediction[i] > 1:
      colorcar[0, 0, i] = 255
    else:
      colorcar[0, 0, i] = prediction[i]*255
  return colorcar

def build_argparser():
  parser = ArgumentParser()
  parser.add_argument('-m', '--model', help='Path to an .xml file with a trained model.', required=True, type=str)
  parser.add_argument('-l', '--cpu_extension',
                      help='MKLDNN (CPU)-targeted custom layers. \
                        Absolute path to a shared library with the kernels implementation',
                      type=str, default=None)
  parser.add_argument('-pp', '--plugin_dir', help='Path to a plugin folder', type=str, default=None)
  parser.add_argument('-d', '--device',
                      help='Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Sample '
                           'will look for a suitable plugin for device specified (CPU by default)', default='CPU',
                      type=str)
  parser.add_argument('--output', '-o', help='Output image')
  parser.add_argument('input_image', help='Image with a vehicle')
  return parser

def load_ir_model(model_xml, device, plugin_dir, cpu_extension):
  model_bin = os.path.splitext(model_xml)[0] + ".bin"

  # initialize plugin
  log.info("Initializing plugin for %s device...", device)
  plugin = IEPlugin(device=device, plugin_dirs=plugin_dir)
  if cpu_extension and 'CPU' in device:
    plugin.add_cpu_extension(cpu_extension)

  # read IR
  net = IENetwork(model=model_xml, weights=model_bin)

  if "CPU" in device:
    supported_layers = plugin.get_supported_layers(net)
    not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
    if not_supported_layers:
      log.error("Following layers are not supported by the plugin for specified device %s:\n %s",
                plugin.device, ', '.join(not_supported_layers))
      log.error("Please try to specify cpu extensions library path in sample's command line parameters using "
                "--cpu_extension command line argument")
      sys.exit(1)

  input_blob = next(iter(net.inputs))
  out_blob = next(iter(net.outputs))
  exec_net = plugin.load(network=net, num_requests=2)
  shape = net.inputs[input_blob].shape  # pylint: disable=E1136
  del net

  return exec_net, plugin, input_blob, out_blob, shape

def main():
  log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
  args = build_argparser().parse_args()

  exec_net, _, input_blob, _, shape = load_ir_model(args.model, args.device, args.plugin_dir, args.cpu_extension)
  net_height, net_width = shape[2:4]

  frame = cv2.imread(args.input_image)
  in_frame = cv2.resize(frame, (net_height, net_width), interpolation=cv2.INTER_CUBIC)
  in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW

  res = exec_net.infer({input_blob: in_frame})
  vtype = CLASS_MAP.get(np.argmax(res['resnet_v1_10/type'][0]), 'undefined')

  colorcar = normalized_to_absolute(res['resnet_v1_10/color'][0])
  rgb_color = cv2.cvtColor(colorcar, cv2.COLOR_LAB2BGR)[0, 0].tolist()

  print("Type: %s" % vtype)
  print("Color: %s" % rgb_color)

  cv2.rectangle(frame, (0, 0), (30, 30), rgb_color, -1)
  font = cv2.FONT_HERSHEY_PLAIN
  cv2.putText(frame, vtype, (0, 15), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

  if args.output:
    cv2.imwrite(args.output, frame)
  else:
    cv2.imshow('Vehicle_attributes', frame)
    cv2.waitKey(0)


if __name__ == '__main__':
  sys.exit(main() or 0)
