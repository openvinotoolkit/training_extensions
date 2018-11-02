import datetime
import json
import multiprocessing as mp
import os
import shutil
import unittest

import cv2
import tensorflow as tf

from eval import eval_once
from export import export
from general.utils import load_module, download_archive_and_extract
from infer import infer
from train import train


cfg = load_module('config.py')
os.environ['MKL_NUM_THREADS'] = '1'
OPEN_VINO_DIR = os.environ.get('OPEN_VINO_DIR', '')

if OPEN_VINO_DIR != '':
  from openvino.inference_engine import IENetwork, IEPlugin
else:
  print('Environment variable OPEN_VINO_DIR is not set')


def download_test_data():
  src_path = 'https://download.01.org/openvinotoolkit/2018_R3/training_toolbox_tensorflow/data/test/test_v1.0.0.zip'
  target_dir = os.path.join(cfg.root_dir, './data/test')

  tf.logging.info('Downloading "{0}" to "{1}"'.format(src_path, target_dir))
  download_archive_and_extract(src_path, target_dir)


class TrainNetCheckRegression(unittest.TestCase):
  def setUp(self):
    tf.logging.set_verbosity(tf.logging.INFO)
    download_test_data()
    self.model_dir = 'model/train_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if not os.path.exists(self.model_dir):
      os.makedirs(self.model_dir)
    else:
      os.rmdir(self.model_dir)
      os.makedirs(self.model_dir)
    cfg.model_dir = self.model_dir

  @staticmethod
  def run_train():
    proc = mp.Process(target=train,
                      args=(cfg,))
    proc.start()
    proc.join()

  @staticmethod
  def run_infer():
    proc = mp.Process(target=infer,
                      args=(cfg, 'json', os.path.join(cfg.root_dir, './data/test/annotations_test.json'), 0.005, True))
    proc.start()
    proc.join()

  @staticmethod
  def run_eval():
    eval_once(cfg, checkpoint=None, save_sample_prediction=True)

  def test_train(self):
    TrainNetCheckRegression.run_train()
    TrainNetCheckRegression.run_infer()
    TrainNetCheckRegression.run_eval()
    if OPEN_VINO_DIR != '':
      self.check_detections(os.path.join(cfg.root_dir, './data/test/openvino_setup/gold_predictions.json'))
      self.check_tf_events(self.model_dir, os.path.join(cfg.root_dir, './data/test/openvino_setup/gold_train_tfevents'))
      self.check_tf_events(os.path.join(self.model_dir, 'eval_test'),
                           os.path.join(cfg.root_dir, './data/test/openvino_setup/gold_eval_tfevents'))
    else:
      self.check_detections(os.path.join(cfg.root_dir, './data/test/default_setup/gold_predictions.json'))
      self.check_tf_events(self.model_dir, os.path.join(cfg.root_dir, './data/test/default_setup/gold_train_tfevents'))
      self.check_tf_events(os.path.join(self.model_dir, 'eval_test'),
                           os.path.join(cfg.root_dir, './data/test/default_setup/gold_eval_tfevents'))

  def check_detections(self, path_to_gold_detections):
    with open(os.path.join(self.model_dir, 'predictions/annotations_test.json')) as file:
      detections = json.load(file)
    with open(path_to_gold_detections) as file:
      gold_detections = json.load(file)

    self.assertEqual(len(gold_detections), len(detections))
    for gold_d, det in zip(gold_detections, detections):
      self.assertEqual(gold_d['image_id'], det['image_id'])
      self.assertEqual(gold_d['category_id'], det['category_id'])
      for val, val_gold in zip(gold_d['bbox'], det['bbox']):
        self.assertAlmostEqual(val, val_gold, msg='Detections mismatch')

  @staticmethod
  def parse_tf_event_file(path_to_tfevent):
    summary = {}
    for event in tf.train.summary_iterator(path_to_tfevent):
      for val in event.summary.value:
        if val.tag in summary:
          summary[val.tag].append(val.simple_value)
        else:
          summary[val.tag] = [val.simple_value]
    return summary

  def check_tf_events(self, tf_event_dir, path_to_gold_tfevent):
    for file in os.listdir(tf_event_dir):
      if 'tfevents' in file:
        event_file = os.path.join(tf_event_dir, file)

    summary = TrainNetCheckRegression.parse_tf_event_file(event_file)
    summary_gold = TrainNetCheckRegression.parse_tf_event_file(path_to_gold_tfevent)

    for tag_gold in summary_gold:
      if tag_gold == 'global_step/sec':  # Don't evaluate performance
        continue
      self.assertTrue(tag_gold in summary, msg='Tag: {0}'.format(tag_gold))
      self.assertEqual(len(summary[tag_gold]), len(summary_gold[tag_gold]), msg='Tag: {0}'.format(tag_gold))
      for val, val_gold in zip(summary[tag_gold], summary_gold[tag_gold]):
        self.assertAlmostEqual(val, val_gold, msg='Tag: {0}'.format(tag_gold))


@unittest.skipIf(OPEN_VINO_DIR == '', 'Environment variable OPEN_VINO_DIR is not set')
class ExportModelCheckRegression(unittest.TestCase):
  def setUp(self):
    download_test_data()
    tf.logging.set_verbosity(tf.logging.INFO)
    self.model_dir = 'model/test_export_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if not os.path.exists(self.model_dir):
      os.makedirs(self.model_dir)
    else:
      os.rmdir(self.model_dir)
      os.makedirs(self.model_dir)
    cfg.model_dir = self.model_dir
    model_ckpt = os.path.join(cfg.root_dir, './data/test/model_ckpt')
    for file in os.listdir(model_ckpt):
      shutil.copy(os.path.join(model_ckpt, file), self.model_dir)

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
                        cfg, os.path.join(OPEN_VINO_DIR, 'deployment_tools/model_optimizer/mo.py'), ))
    proc.start()
    proc.join()

  def compare_ie_models(self):

    ie_detection = ExportModelCheckRegression.run_ie_on_dataset(
      os.path.join(cfg.model_dir, 'ie_model/graph.xml'),
      os.path.join(cfg.model_dir, 'ie_model/graph.bin'),
      os.path.join(OPEN_VINO_DIR,
                   'deployment_tools/inference_engine/lib/ubuntu_16.04/intel64/libcpu_extension_avx2.so'),
      os.path.join(cfg.root_dir, './data/test/test'))

    ie_gold_detection = self.run_ie_on_dataset(
      os.path.join(cfg.root_dir, './data/test/graph_gold.xml'),
      os.path.join(cfg.root_dir, './data/test/graph_gold.bin'),
      os.path.join(OPEN_VINO_DIR,
                   'deployment_tools/inference_engine/lib/ubuntu_16.04/intel64/libcpu_extension_avx2.so'),
      os.path.join(cfg.root_dir, './data/test/test'))

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


if __name__ == '__main__':
  unittest.main()
