from __future__ import print_function
import os
import shutil
import unittest
import multiprocessing as mp

import cv2
import tensorflow as tf

from ssd_detector.export import export
from tests.ssd_detector import OPEN_VINO_DIR, CONFIG
from tests.ssd_detector.base_test import BaseTest

if OPEN_VINO_DIR != '':
  from openvino.inference_engine import IENetwork, IEPlugin
else:
  print('Environment variable OPEN_VINO_DIR is not set')


@unittest.skipIf(OPEN_VINO_DIR == '', 'Environment variable OPEN_VINO_DIR is not set')
class ExportModelCheckRegression(BaseTest):
  def setUp(self):
    tf.logging.set_verbosity(tf.logging.INFO)
    self.create_model_dir('export')
    model_ckpt = os.path.join(CONFIG.ROOT_DIR, 'data/test/model_ckpt')
    for file in os.listdir(model_ckpt):
      shutil.copy(os.path.join(model_ckpt, file), CONFIG.MODEL_DIR)

  # pylint: disable=too-many-locals
  @staticmethod
  def run_ie_on_dataset(model_xml, model_bin, cpu_extension_path, images_dir, prob_threshold=0.01):
    plugin = IEPlugin(device='CPU')
    plugin.add_cpu_extension(cpu_extension_path)
    net = IENetwork.from_ir(model=model_xml, weights=model_bin)
    assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    exec_net = plugin.load(network=net, num_requests=2)
    num, chs, height, width = net.inputs[input_blob]
    del net
    cur_request_id = 0

    detection_data = []
    for image in os.listdir(images_dir):
      im_path = os.path.join(images_dir, image)
      frame = cv2.imread(im_path)
      initial_h, initial_w, _ = frame.shape
      in_frame = cv2.resize(frame, (width, height))
      in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
      in_frame = in_frame.reshape((num, chs, height, width))

      objects_per_image = []
      exec_net.start_async(request_id=cur_request_id, inputs={input_blob: in_frame})

      if exec_net.requests[cur_request_id].wait(-1) == 0:
        res = exec_net.requests[cur_request_id].outputs[out_blob]
        for obj in res[0][0]:
          if obj[2] > prob_threshold:
            xmin = int(obj[3] * initial_w)
            ymin = int(obj[4] * initial_h)
            xmax = int(obj[5] * initial_w)
            ymax = int(obj[6] * initial_h)
            class_id = int(obj[1])
            conf = obj[2]
            objects_per_image.append({'bbox': [xmin, ymin, xmax, ymax], 'class_id': class_id, 'score': conf})

      det_item = {'image': im_path, 'objects': objects_per_image}
      detection_data.append(det_item)

    del exec_net
    del plugin

    return detection_data

  @staticmethod
  def run_export():
    proc = mp.Process(target=export,
                      args=(
                        CONFIG, os.path.join(OPEN_VINO_DIR, 'deployment_tools/model_optimizer/mo.py'), ))
    proc.start()
    proc.join()

  def compare_ie_models(self):

    ie_detection = ExportModelCheckRegression.run_ie_on_dataset(
      os.path.join(CONFIG.MODEL_DIR, 'ie_model/graph.xml'),
      os.path.join(CONFIG.MODEL_DIR, 'ie_model/graph.bin'),
      os.path.join(OPEN_VINO_DIR,
                   'deployment_tools/inference_engine/lib/ubuntu_16.04/intel64/libcpu_extension_avx2.so'),
      os.path.join(CONFIG.ROOT_DIR, 'data/test/test'))

    ie_gold_detection = self.run_ie_on_dataset(
      os.path.join(CONFIG.ROOT_DIR, 'data/test/graph_gold.xml'),
      os.path.join(CONFIG.ROOT_DIR, 'data/test/graph_gold.bin'),
      os.path.join(OPEN_VINO_DIR,
                   'deployment_tools/inference_engine/lib/ubuntu_16.04/intel64/libcpu_extension_avx2.so'),
      os.path.join(CONFIG.ROOT_DIR, 'data/test/test'))

    self.assertEqual(len(ie_detection), len(ie_gold_detection))
    for gold_d, det in zip(ie_gold_detection, ie_detection):
      self.assertEqual(gold_d['image'], det['image'])
      self.assertEqual(len(gold_d['objects']), len(det['objects']))
      for obj, obj_gold in zip(det['objects'], gold_d['objects']):
        self.assertEqual(obj['class_id'], obj_gold['class_id'])
        self.assertAlmostEqual(obj['score'], obj_gold['score'])
        for val, val_gold in zip(obj['bbox'], obj_gold['bbox']):
          self.assertAlmostEqual(val, val_gold, msg='Detections mismatch')

  def test_export(self):
    ExportModelCheckRegression.run_export()
    self.compare_ie_models()
