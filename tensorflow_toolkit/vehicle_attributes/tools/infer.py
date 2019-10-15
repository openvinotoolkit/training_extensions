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

CLASS_MAP = {
  0: 'car',
  1: 'bus',
  2: 'track',
  3: 'van',
}

def load_graph(frozen_graph_filename):
  with tf.gfile.GFile(frozen_graph_filename, 'rb') as file:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(file.read())
  with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def)
  return graph

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
  parser.add_argument('--model', '-m', help='Path to frozen graph file with a trained model.', required=True, type=str)
  parser.add_argument('--output', '-o', help='Output image')
  parser.add_argument('input_image', help='Image with license plate')
  return parser


def main():
  args = build_argparser().parse_args()

  graph = load_graph(args.model)

  image = cv2.imread(args.input_image)
  img = cv2.resize(image, (72, 72))
  img = np.float32(img)
  img = np.multiply(img, 1.0/255.0)

  input = graph.get_tensor_by_name("import/input:0")
  output = [
    graph.get_tensor_by_name("import/resnet_v1_10/type:0"),
    graph.get_tensor_by_name("import/resnet_v1_10/color:0")
  ]

  with tf.Session(graph=graph) as sess:
    results = sess.run(output, feed_dict={input: [img]})

    vtype = CLASS_MAP.get(np.argmax(results[0][0]), 'undefined')
    colorcar = normalized_to_absolute(results[1][0])
    rgb_color = cv2.cvtColor(colorcar, cv2.COLOR_LAB2BGR)[0, 0].tolist()

    print("Type: %s" % vtype)
    print("Color: %s" % rgb_color)

    cv2.rectangle(image, (0, 0), (30, 30), rgb_color, -1)
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(image, vtype, (0, 15), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

    if args.output:
      cv2.imwrite(args.output, image)
    else:
      cv2.imshow('Vehicle_attributes', image)
      cv2.waitKey(0)


if __name__ == "__main__":
  main()
