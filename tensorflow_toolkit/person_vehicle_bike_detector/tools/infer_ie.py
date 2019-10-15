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

from argparse import ArgumentParser
import os
import sys
import logging as log
import numpy as np
import cv2
from object_detection.utils import label_map_util
from openvino.inference_engine import IENetwork, IEPlugin


def build_argparser():
  parser = ArgumentParser()
  parser.add_argument("--model", help="Path to frozen graph", required=True, type=str)
  parser.add_argument("--cpu_extension",
                      help="MKLDNN (CPU)-targeted custom layers. "
                           "Absolute path to a shared library with the kernels implementation", type=str, default=None)
  parser.add_argument("--plugin_dir", help="Path to a plugin folder", type=str, default=None)
  parser.add_argument("--device",
                      help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                           "will look for a suitable plugin for device specified (CPU by default)", default="CPU",
                      type=str)
  parser.add_argument("--label_map", help="Path to frozen graph", default="dataset/crossroad_label_map.pbtxt", type=str)
  parser.add_argument('--output', '-o', help='Output image')
  parser.add_argument('input_image', help='Image with license plate')
  return parser.parse_args()


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


def draw(image, detections, label_map, conf_threshold=0.25, bbox_color=(50, 255, 50)):
  height, width = image.shape[:2]
  for obj in detections:
    _, class_id, score, xmin, ymin, xmax, ymax = obj
    class_id = int(class_id)
    if score > conf_threshold:
      xmin = int(xmin * width)
      ymin = int(ymin * height)
      xmax = int(xmax * width)
      ymax = int(ymax * height)
      label = "{0}: {1:.2f}".format(label_map[class_id]['name'], score)
      cv2.rectangle(image, (xmin, ymin), (xmax, ymax), bbox_color, 2)
      label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
      cv2.rectangle(image, (xmin, ymin-label_size[0][1]), (xmin+label_size[0][0], ymin+label_size[1]),
                    (255, 255, 255), cv2.FILLED)
      cv2.putText(image, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)


def main():
  args = build_argparser()

  label_map = label_map_util.create_category_index_from_labelmap(args.label_map, use_display_name=True)
  exec_net, _, input_blob, _, shape = load_ir_model(args.model, args.device, args.plugin_dir, args.cpu_extension)
  net_height, net_width = shape[2:4]
  image = cv2.imread(args.input_image)
  img = cv2.resize(image, (net_width, net_height))
  img = np.float32(img)
  # Change data layout from HWC to CHW
  img = img.transpose((2, 0, 1))  # pylint: disable=E1111,E1121
  img = img.reshape(shape)
  res = exec_net.infer(inputs={input_blob: img})

  detections = res['DetectionOutput'][0][0]

  draw(image, detections, label_map)

  if args.output:
    cv2.imwrite(args.output, image)
  else:
    cv2.imshow('Image', image)
    cv2.waitKey(0)

if __name__ == "__main__":
  main()
