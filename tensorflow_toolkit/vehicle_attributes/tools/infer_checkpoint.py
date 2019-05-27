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

import random
import json
import argparse
import numpy as np

import cv2
import tensorflow as tf

from colormath.color_diff import delta_e_cie1976
from colormath.color_objects import LabColor

from tfutils.helpers import load_module
from vehicle_attributes.trainer import create_session, resnet_v1_10_1
from vehicle_attributes.readers.vehicle_attributes_json import BarrierAttributesJson

def parse_args():
  parser = argparse.ArgumentParser(description='Perform inference of vehicle attributes model')
  parser.add_argument('path_to_config', help='Path to a config.py')
  return parser.parse_args()

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

# pylint: disable=too-many-locals, too-many-statements, invalid-name, too-many-boolean-expressions, len-as-condition
def infer(config):
  session_config = create_session(config, 'infer')

  run_config = tf.estimator.RunConfig(session_config=session_config)

  va_estimator = tf.estimator.Estimator(
    model_fn=resnet_v1_10_1,
    params=config.resnet_params,
    model_dir=config.model_dir,
    config=run_config)

  with open(config.infer.annotation_path) as f:
    data = json.load(f)
    pic = 0
    summ = 0
    random.seed(666)
    for _ in range(len(data)):
      pic = random.randint(0, len(data)-1)
      images, annotations = BarrierAttributesJson.get_annotations(data[pic])
      if len(images) == 0:
        pic += 1
        continue
      print("pic = ", pic)
      predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=np.array([images], dtype=np.float32).reshape([-1] + list(config.input_shape)),
        num_epochs=1,
        shuffle=False)
      predict = va_estimator.predict(input_fn=predict_input_fn)

      img = cv2.imread(data[pic]['image'], -1)
      cv2.namedWindow("example")

      cars = []

      for y in range(len(data[pic]['objects'])):
        if data[pic]['objects'][y]['label'] == 'vehicle' and 'bbox' in data[pic]['objects'][y] and \
          len(data[pic]['objects'][y]['bbox']) != 0 and 'color_bbox' in data[pic]['objects'][y]['attributes'] and \
            len(data[pic]['objects'][y]['attributes']['color_bbox']) != 0 and \
              'type' in data[pic]['objects'][y]['attributes']:
          cars.append(y)
      it = 0
      summ_temp = 0
      for i in predict:
        colorcar = normalized_to_absolute(i['color_lab'])
        n = cars[it]

        bbox_car = data[pic]['objects'][n]['bbox']
        color_detected = LabColor(colorcar[0][0][0], colorcar[0][0][1], colorcar[0][0][2])
        colorcar_rgb = cv2.cvtColor(colorcar, cv2.COLOR_LAB2BGR)[0, 0].tolist()
        cv2.rectangle(img, (int(bbox_car[0]), int(bbox_car[1])), (int(bbox_car[2]), int(bbox_car[3])),
                      colorcar_rgb,
                      thickness=5)

        l2diss = 10000
        tempy = 0

        for j, item in enumerate(annotations):
          colorcar_given = normalized_to_absolute(item[4:7])
          color_given = LabColor(colorcar_given[0][0][0], colorcar_given[0][0][1], colorcar_given[0][0][2])
          l2diss_temp = delta_e_cie1976(color_given, color_detected)
          if l2diss_temp <= l2diss:
            l2diss = l2diss_temp
            tempy = j
        colorcar_given = normalized_to_absolute(annotations[tempy][4:7])
        colorcar_given_rgb = cv2.cvtColor(colorcar_given, cv2.COLOR_LAB2BGR)[0, 0].tolist()
        color_given = LabColor(colorcar_given[0][0][0], colorcar_given[0][0][1], colorcar_given[0][0][2])
        l2diss = delta_e_cie1976(color_given, color_detected)

        vtype = BarrierAttributesJson.one_hot_annotation_to_type(i['types_class'])
        gttype = data[pic]['objects'][n]['attributes']['type']
        if gttype in ('suv', 'mpv', 'other'):
          gttype = 'car'
        if gttype == 'pickup':
          gttype = 'truck'

        overlay = img.copy()
        cv2.rectangle(img, (0, 0 + 60 * it), (120, 60 + 60 * it), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.3, img, 1.0 - 0.3, 0.0, img)
        cv2.rectangle(img, (5, 5 + 60 * it), (15, 15 + 60 * it), colorcar_rgb, -1)
        cv2.rectangle(img, (5, 25 + 60 * it), (15, 35 + 60 * it), colorcar_given_rgb, -1)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(img, vtype, (35, 15 + 60 * it), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, gttype + " gt", (35, 35 + 60 * it), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        it += 1
        summ_temp += l2diss
      summ_temp /= it
      summ += summ_temp
      pic += 1
      cv2.imshow("example", img)
      press = cv2.waitKey(0)
      if press == 27:
        break
  summ /= len(data)
  print(summ)

def main(_):
  args = parse_args()
  cfg = load_module(args.path_to_config)
  infer(cfg)

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
