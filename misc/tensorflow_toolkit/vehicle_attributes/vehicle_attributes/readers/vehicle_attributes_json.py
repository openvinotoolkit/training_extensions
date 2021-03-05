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

"""
  Dataset reader.
"""

import json
import os

import sys
import threading
from tqdm import tqdm

import cv2
import jpeg4py as jpeg
import numpy as np
import tensorflow as tf

def imread(im_path):
  if os.path.splitext(im_path)[1].lower() in ('.jpg', '.jpeg'):
    try:
      img = jpeg.JPEG(im_path).decode()[..., ::-1]  # RGB -> BGR
    # pylint: disable=broad-except
    except Exception as ex:
      tf.logging.warning("Can't load {0} with jpeg4py (libjpeg-turbo): {1}. Will use OpenCV.".format(im_path, ex))
      img = cv2.imread(im_path, cv2.IMREAD_COLOR)
  else:
    img = cv2.imread(im_path, cv2.IMREAD_COLOR)
  return img


def imdecode(data):
  try:
    img = jpeg.JPEG(data).decode()[..., ::-1]  # RGB -> BGR
  # pylint: disable=broad-except
  except Exception as ex:
    tf.logging.warning("Can't decode with jpeg4py (libjpeg-turbo): {0}. Will use OpenCV.".format(ex))
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
  return img


class BarrierAttributesJson:
  _cache = dict()
  _lock = threading.RLock()

  @staticmethod
  def json_iterator(size):
    def generator():
      for i in range(size):
        yield i

    return generator

  # pylint: disable=invalid-name
  @staticmethod
  def init_cache(filename, cache_type):
    print('Load images in the cache: {}'.format(cache_type))

    with open(filename) as f:
      items = json.load(f)

    size = len(items)

    id = 0

    with BarrierAttributesJson._lock:
      t = tqdm(items, total=size, unit='images')
      total_cache_usage = 0
      for item in t:
        images, annotations = BarrierAttributesJson._get_annotation(item)

        for image, annotation in zip(images, annotations):
          BarrierAttributesJson._cache[id] = [image, annotation]
          id += 1

          total_cache_usage += sys.getsizeof(image)
          total_cache_usage += sys.getsizeof(annotation)

        t.set_postfix({'cache usage (GB)': total_cache_usage / 1024 ** 3})

    return id

  @staticmethod
  def _get_image_and_annotation(item):
    image, annotation = BarrierAttributesJson._cache[item]
    return image, annotation

  @staticmethod
  def type_annotation_to_one_hot(item):
    vtype = np.zeros(4)
    if item in ('car', 'mpv', 'suv', 'other'):
      vtype[0] = 1 #car
    elif item == 'bus':
      vtype[1] = 1 #bus
    elif item in ('pickup', 'truck'):
      vtype[2] = 1 #truck
    elif item == 'van':
      vtype[3] = 1 #van
    else:
      print('Error of type recognition')
    return vtype

  @staticmethod
  def one_hot_annotation_to_type(t):
    if t[0] == 1:
      vehtype = 'car'
    elif t[1] == 1:
      vehtype = 'bus'
    elif t[2] == 1:
      vehtype = 'truck'
    elif t[3] == 1:
      vehtype = 'van'
    else:
      vehtype = 'undefined'
    return vehtype

  # pylint: disable=len-as-condition
  @staticmethod
  def _get_annotation(frame):
    images = []
    annotations = []
    for item in frame['objects']:
      if item['label'] == 'vehicle' and \
        'color_bbox' in item['attributes'] and \
        len(item['attributes']['color_bbox']) != 0 and \
        'type' in item['attributes'] and \
        len(item['attributes']['type']) != 0:
        img = imread(frame['image'])

        vbbox = item['bbox']
        veh_image = img[int(vbbox[1]):int(vbbox[3]), int(vbbox[0]): int(vbbox[2])]
        veh_image = cv2.resize(veh_image, (72, 72), interpolation=cv2.INTER_CUBIC)

        #get color annotation
        veh_color = np.zeros(3)
        bbox = item['attributes']['color_bbox'][0]
        if len(item['attributes']['color_bbox']) == 2:
          if item['attributes']['color_bbox'][0][1] <= \
            item['attributes']['color_bbox'][1][1]:
            bbox = item['attributes']['color_bbox'][0]
          else:
            bbox = item['attributes']['color_bbox'][1]

        roi_color = img[int(bbox[1]):int(bbox[3]), int(bbox[0]): int(bbox[2])]
        roi_color = cv2.cvtColor(roi_color, cv2.COLOR_BGR2LAB)
        veh_color[0:3] = np.mean(roi_color.reshape(-1, 3), axis=0)

        #get type annotation
        veh_type = BarrierAttributesJson.type_annotation_to_one_hot(item['attributes']['type'])

        images.append(veh_image.astype(np.float32) / 255.)
        anno = np.concatenate((veh_type, veh_color.astype(np.float32) / 255.))
        annotations.append(anno)

    return images, annotations

  @staticmethod
  def json_decode_entry(value):
    img, annotation = BarrierAttributesJson._get_image_and_annotation(value)
    return img, annotation

  @staticmethod
  def create_dataset(size):
    gen = BarrierAttributesJson.json_iterator(size)
    return tf.data.Dataset.from_generator(gen, tf.int64, tf.TensorShape([]))

  @staticmethod
  def transform_fn(value):
    image, annotation = BarrierAttributesJson.json_decode_entry(value)
    result = image.astype(np.float32), annotation.astype(np.float32)
    return result

  @staticmethod
  def get_annotations(item):
    imgs, annotations = BarrierAttributesJson._get_annotation(item)
    return imgs, annotations
