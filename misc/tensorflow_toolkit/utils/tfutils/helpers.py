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
from importlib import util
from os import path, system
import sys

import subprocess
import numpy as np
import tensorflow as tf
import cv2


def import_research_models():
  research_dir = path.realpath(path.join(path.dirname(__file__), '../../../../external/models/research/'))
  assert path.exists(research_dir), f'Does not exist: {research_dir}'
  sys.path.append(research_dir)
  assert path.exists(path.join(research_dir, 'slim')), f'Does not exist: {path.join(research_dir, "slim")}'
  sys.path.append(path.join(research_dir, 'slim'))


def import_transformer():
  transformer_dir = path.realpath(path.join(path.dirname(__file__), '../../../../external/models/research/transformer'))
  assert path.exists(transformer_dir), f'Does not exist: {transformer_dir}'
  sys.path.append(transformer_dir)


def load_module(module_name):
  # TODO: replace on
  # __import__(module_name)
  # return sys.modules[module_name]
  spec = util.spec_from_file_location("module.name", module_name)
  module = util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


# pylint: disable=too-many-locals
def draw_bboxes(val_images, annotations, predictions, classes, conf_threshold=0.5):
  font = cv2.FONT_HERSHEY_TRIPLEX
  font_scale = 0.6
  font_thickness = 1
  alpha = 0.5
  rect_thickness = 2

  images = []

  for im_idx, _ in enumerate(val_images):
    img = val_images[im_idx].copy()
    height, width = img.shape[:2]

    annotation = annotations[im_idx]

    for _, bboxes in annotation.items():
      for bbox in bboxes:
        top_left = int(round(bbox.xmin * width)), int(round(bbox.ymin * height))
        bottom_right = int(round(bbox.xmax * width)), int(round(bbox.ymax * height))
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), thickness=rect_thickness)

    det_label = predictions[im_idx][:, 1]
    det_conf = predictions[im_idx][:, 2]
    det_xmin = predictions[im_idx][:, 3]
    det_ymin = predictions[im_idx][:, 4]
    det_xmax = predictions[im_idx][:, 5]
    det_ymax = predictions[im_idx][:, 6]

    order = np.argsort(np.array(det_conf))[::-1]

    for bb_idx in order:
      score = det_conf[bb_idx]

      if score == 0:
        break

      if score < conf_threshold:  # last
        color = (128, 0, 128)
      else:
        color = (255, 0, 0)

      top_left = int(round(det_xmin[bb_idx] * width)), int(round(det_ymin[bb_idx] * height))
      bottom_right = int(round(det_xmax[bb_idx] * width)), int(round(det_ymax[bb_idx] * height))

      top_left = tuple(np.clip(top_left, (0, 0), (width - 1, height - 1)))
      bottom_right = tuple(np.clip(bottom_right, (0, 0), (width - 1, height - 1)))
      cv2.rectangle(img, top_left, bottom_right, color, thickness=rect_thickness)

      label = classes[int(det_label[bb_idx])] if classes else int(det_label[bb_idx])

      display_txt = '{0}:{1:0.2f}'.format(label, score)

      txt_size = cv2.getTextSize(display_txt, font, font_scale, font_thickness)[0]

      # Fill background for text with alpha-blending
      text_tl = (top_left[0], top_left[1])
      text_tl = tuple(np.clip(text_tl, (0, 0), (width - 1, height - 1)))
      text_br = (top_left[0] + txt_size[0], top_left[1] + txt_size[1])
      text_br = tuple(np.clip(text_br, (0, 0), (width - 1, height - 1)))
      roi = img[text_tl[1]:text_br[1], text_tl[0]:text_br[0]].astype(np.float32)
      roi *= alpha
      roi += (1. - alpha) * 255
      img[text_tl[1]:text_br[1], text_tl[0]:text_br[0]] = roi.astype(np.uint8)
      cv2.putText(img, display_txt, (text_tl[0], text_tl[1] + txt_size[1]), font, font_scale, (0, 128, 0),
                  font_thickness)

      if score < conf_threshold:
        break

    images.append(img)

  return images


def estimate_inputs_outputs(graph):
  unlikely_output = ['Const', 'Assign', 'NoOp', 'Placeholder', 'Assert',
                     'switch_t', 'switch_f', 'IsVariableInitialized', 'Save', 'SaveV2']
  outputs = []
  inputs = []
  for node in graph.as_graph_def().node:
    if node.op == 'Placeholder':
      inputs.append(node.name)

    if node.op not in unlikely_output:
      if node.name.split('/')[-1] not in unlikely_output:
        operation = graph.get_operation_by_name(node.name)

        children_count = sum(1 for out in operation.outputs for _ in out.consumers())
        if children_count == 0:
          outputs.append(node.name)

  return inputs, outputs


def dump_frozen_graph(sess, graph_file, output_node_names=None):
  assert graph_file.endswith('.pb')
  assert output_node_names is None or isinstance(output_node_names, list)
  output_node_names = output_node_names or estimate_inputs_outputs(sess.graph)[1]

  dir_ = path.dirname(graph_file)
  base = path.basename(graph_file)
  ckpt = graph_file.replace('.pb', '.ckpt')
  frozen = graph_file.replace('.pb', '.pb.frozen')

  system('mkdir -p {}'.format(dir_))
  print('>> Saving `{}`... '.format(graph_file))
  tf.train.write_graph(sess.graph, dir_, base, as_text=False)
  tf.train.write_graph(sess.graph, dir_, base + "txt", as_text=True)
  print('Done')

  print('>> Saving `{}`... '.format(ckpt))
  with sess.graph.as_default():
    saver = tf.train.Saver()
    saver.save(sess, ckpt, write_meta_graph=False)
  print('Done')

  print('>> Freezing graph to `{}`... '.format(frozen))
  print('Outputs:\n  {}'.format(', '.join(output_node_names)))

  from tensorflow.python.tools.freeze_graph import freeze_graph
  freeze_graph(input_graph=graph_file,
               input_saver='',
               input_binary=True,
               input_checkpoint=ckpt,
               output_node_names=','.join(output_node_names),
               restore_op_name='save/restore_all',
               filename_tensor_name='save/Const:0',
               output_graph=frozen,
               clear_devices=True,
               initializer_nodes='',
               saved_model_tags='serve')

  return frozen


def download_archive_and_extract(url, target_dir):
  from io import BytesIO
  from zipfile import ZipFile
  from urllib.request import urlopen
  import urllib

  try:
    resp = urlopen(url)
  except urllib.error.HTTPError as exception:
    tf.logging.error('Not found: {}'.format(url))
    raise exception

  zipfile = ZipFile(BytesIO(resp.read()))
  zipfile.extractall(target_dir)


def execute_mo(config, frozen, output_dir):
  command = [
    'mo.py',
    '--framework=tf',
    '--input_model={}'.format(frozen),
    '--output_dir={}'.format(output_dir),
  ]

  for arg, value in config.items():
    if not value:
      continue
    if isinstance(value, bool):
        command.append('--{}'.format(arg))
    elif isinstance(value, (str, int, float)):
      command.append('--{}={}'.format(arg, value))
    elif isinstance(value, list):
      command.append('--{}=[{}]'.format(arg, ','.join([str(x) for x in value])))
    else:
      raise Exception('Unexpected format of value in mo_config: {}'.format(value))

  print('')
  print(' '.join(command))

  subprocess.call(command)
