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
import tensorflow as tf
import numpy as np
import cv2

from tfutils.helpers import draw_bboxes


def load_graph(frozen_graph_filename):
  with tf.gfile.GFile(frozen_graph_filename, 'rb') as file:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(file.read())
  with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def)
  return graph

def display_license_plate(number, license_plate_img):
  size = cv2.getTextSize(number, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
  text_width = size[0][0]
  text_height = size[0][1]

  height, width, _ = license_plate_img.shape
  license_plate_img = cv2.copyMakeBorder(license_plate_img, 0, text_height + 10, 0,
                                         0 if text_width < width else text_width - width,
                                         cv2.BORDER_CONSTANT, value=(255, 255, 255))
  cv2.putText(license_plate_img, number, (0, height + text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

  return license_plate_img

def build_argparser():
  parser = ArgumentParser()
  parser.add_argument('--model', help='Path to frozen graph file with a trained model.', required=True, type=str)
  parser.add_argument('input_image', help='Image with license plate')
  return parser


def main():
  args = build_argparser().parse_args()

  graph = load_graph(args.model)
  image = cv2.imread(args.input_image)

  img = cv2.resize(image, (256, 256))
  img = np.float32(img)
  img = np.add(img, -127.5)
  img = np.multiply(img, 1.0/127.5)

  input = graph.get_tensor_by_name("import/MobilenetV2/input:0")
  output = graph.get_tensor_by_name("import/DetectionOutput/Reshape:0")

  with tf.Session(graph=graph) as sess:
    results = sess.run(output, feed_dict={input: [img]})
    img_to_display = draw_bboxes([image], [{}], results, ['', 'vehicle', 'lp'])[0]
    cv2.imshow('', img_to_display)
    cv2.waitKey(0)

if __name__ == "__main__":
  main()
