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

from __future__ import print_function
import os
import pickle
import sys

import cv2
import jpeg4py as jpeg
import matplotlib; matplotlib.use('Agg')  # pylint: disable=multiple-statements
import numpy as np
from pycocotools.coco import COCO
import tensorflow as tf
from tqdm import tqdm

from ssd_detector.toolbox.bounding_box import BoundingBox


def imread(im_path):
  if os.path.splitext(im_path)[1].lower() in ('.jpg', '.jpeg'):
    try:
      img = jpeg.JPEG(im_path).decode()[..., ::-1]  # RGB -> BGR
    # pylint: disable=broad-except
    except Exception as ex:
      tf.logging.warning("Can't load {0} with jpeg4py (libjpeg-turbo): {1}. Will use OpenCV. "
                         "Can be slower.".format(im_path, ex))
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


class ObjectDetectorJson:
  _cache = dict()

  # pylint: disable=invalid-name
  @staticmethod
  def get_classes_from_coco_annotation(ann_path):
    annotation = COCO(ann_path)
    max_id = max(annotation.cats.keys())
    classes = ['None#{}'.format(i) for i in range(max_id + 1)]
    for class_id, class_item in annotation.cats.items():
      classes[class_id] = class_item['name']
    return classes

  # pylint: disable=too-many-locals
  @staticmethod
  def convert_coco_to_toolbox_format(coco_annotation, classes, annotation_directory=''):
    annotations = list(coco_annotation.anns.values())
    converted_annotations = {}
    images = list(coco_annotation.imgs.values())

    images_id_to_name_map = {}
    for image in images:
      image_id = image['id']
      images_id_to_name_map[image_id] = image

      image = images_id_to_name_map[image_id]
      im_path = image['image']
      if not os.path.isabs(im_path):
        im_path = os.path.join(annotation_directory, im_path)
      image_size = [image['width'], image['height']]

      converted_annotations[image_id] = {'image_id': image_id,
                                         'image': im_path,
                                         'image_size': image_size,
                                         'dataset': image['dataset'] if 'dataset' in image else 'DATASET',
                                         'objects': []
                                        }

    for annotation in annotations:
      xmin, ymin, width, height = annotation['bbox']
      xmax = xmin + width
      ymax = ymin + height

      obj = {}
      image_id = annotation['image_id']
      obj['bbox'] = [xmin, ymin, xmax, ymax]
      obj['label'] = classes[annotation['category_id']]
      obj['occluded'] = annotation['is_occluded'] if 'is_occluded' in annotation else False
      obj['attributes'] = annotation['attributes'] if 'attributes' in annotation else dict()

      if image_id in converted_annotations:
        converted_annotations[image_id]['objects'].append(obj)
      else:
        tf.logging.error('Image with image_id = {} is absent, but was found in annotation'.format(image_id))

    images_without_annotation = [key for key, val in converted_annotations.items() if len(val['objects']) == 0]
    tf.logging.info('Images without annotation: {}'.format(len(images_without_annotation)))
    tf.logging.info(images_without_annotation)

    return list(converted_annotations.values())

  @staticmethod
  def json_iterator(filename, classes):
    coco_annotation = COCO(filename)
    annotation_directory = os.path.join(os.getcwd(), os.path.dirname(filename))
    converted_data = ObjectDetectorJson.convert_coco_to_toolbox_format(coco_annotation, classes, annotation_directory)

    def generator():
      for item in converted_data:
        yield pickle.dumps(item)

    return generator, len(converted_data)

  @staticmethod
  def init_cache(filename, cache_type, classes):
    assert cache_type in ('FULL', 'ENCODED', 'NONE')
    print('Load images in the cache: {}'.format(cache_type))
    generator, size = ObjectDetectorJson.json_iterator(filename, classes)
    items = [pickle.loads(item) for item in generator()]

    def _read_image_from_disk(im_path, cache_type):
      assert cache_type in ('ENCODED', 'FULL')
      if cache_type == 'ENCODED':
        with open(im_path, 'rb') as file:
          encoded_image = file.read()
          encoded_image = np.array(bytearray(encoded_image), dtype=np.uint8)
        return encoded_image
      if cache_type == 'FULL':
        image = imread(im_path)
        return image

      return None

    items = tqdm(items, total=size, unit='images')
    total_cache_usage = 0
    for item in items:
      im_path = item['image']
      if cache_type != 'NONE':
        image = _read_image_from_disk(im_path, cache_type)
      else:
        image = None

      annotation = ObjectDetectorJson._get_annotation(item, classes)
      ObjectDetectorJson._cache[im_path] = [image, annotation]

      if isinstance(image, np.ndarray):
        total_cache_usage += image.nbytes
      else:
        total_cache_usage += sys.getsizeof(image)
      total_cache_usage += sys.getsizeof(annotation)  # Bad estimation

      items.set_postfix({'cache usage (GB)': total_cache_usage / 1024 ** 3})

  @staticmethod
  def _get_image_and_annotation(item, cache_type):
    im_path = item['image']

    if im_path not in ObjectDetectorJson._cache:
      tf.logging.error("Image '{0}' is absent in the cache. Wrong path.".format(im_path))
      exit(1)

    if cache_type == 'NONE':
      _, annotation = ObjectDetectorJson._cache[im_path]
      img = imread(im_path)
    else:
      img, annotation = ObjectDetectorJson._cache[im_path]

      if cache_type == 'ENCODED':
        img = imdecode(img)

    return img, annotation

  @staticmethod
  def _get_annotation(item, classes):
    # Annotation is encoded in the original image size, actual image size may be smaller due to performance reasons
    width, height = item['image_size']
    width, height = float(width), float(height)

    annotation = {}
    for obj in item['objects']:
      class_name = obj['label']
      if class_name == 'ignored':
        continue
      try:
        class_id = classes.index(class_name)
      except ValueError:
        tf.logging.warning("Unknown label = '{0}', supported labels: {1}".format(class_name, classes))
        class_id = -1
        continue

      xmin, ymin, xmax, ymax = obj['bbox']

      xmin /= width
      ymin /= height
      xmax /= width
      ymax /= height

      annotation.setdefault(class_id, []).append(BoundingBox(xmin, ymin, xmax, ymax))

    return annotation

  @staticmethod
  def json_decode_entry(value, cache_type):
    value = pickle.loads(value)
    img, annotation = ObjectDetectorJson._get_image_and_annotation(value, cache_type)
    return img, annotation

  @staticmethod
  def create_dataset(json_file_path, classes):
    gen, num = ObjectDetectorJson.json_iterator(json_file_path, classes)
    return tf.data.Dataset.from_generator(gen, tf.string, tf.TensorShape([])), num

  @staticmethod
  def transform_fn(value, transformer, cache_type='NONE', add_original_image=False):
    image, annotation = ObjectDetectorJson.json_decode_entry(value, cache_type)
    transformed_image, annotation = transformer.transform(image, annotation)
    result = transformed_image.astype(np.float32), pickle.dumps(annotation)

    if add_original_image:
      result += (image,)

    return result
