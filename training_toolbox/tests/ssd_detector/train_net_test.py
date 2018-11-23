import json
import multiprocessing as mp
import os

import tensorflow as tf

from ssd_detector.eval import eval_once
from ssd_detector.infer import infer
from ssd_detector.train import train
from tests.ssd_detector import CONFIG, OPEN_VINO_DIR
from tests.ssd_detector.base_test import BaseTest


class TrainNetCheckRegression(BaseTest):
  def setUp(self):
    tf.logging.set_verbosity(tf.logging.INFO)
    self.create_model_dir('train')

  @staticmethod
  def run_train():
    proc = mp.Process(target=train,
                      args=(CONFIG,))
    proc.start()
    proc.join()

  @staticmethod
  def run_infer():
    proc = mp.Process(target=infer,
                      args=(CONFIG, 'json', os.path.join(CONFIG.root_dir, 'data/test/annotations_test.json'), 0.005, True))
    proc.start()
    proc.join()

  @staticmethod
  def run_eval():
    eval_once(CONFIG, checkpoint=None, save_sample_prediction=True)

  def test_train(self):
    TrainNetCheckRegression.run_train()
    TrainNetCheckRegression.run_infer()
    TrainNetCheckRegression.run_eval()
    if OPEN_VINO_DIR != '':
      self.check_detections(os.path.join(CONFIG.root_dir, 'data/test/openvino_setup/gold_predictions.json'))
      self.check_tf_events(CONFIG.model_dir, os.path.join(CONFIG.root_dir, 'data/test/openvino_setup/gold_train_tfevents'))
      self.check_tf_events(os.path.join(CONFIG.model_dir, 'eval_test'),
                           os.path.join(CONFIG.root_dir, 'data/test/openvino_setup/gold_eval_tfevents'))
    else:
      self.check_detections(os.path.join(CONFIG.root_dir, 'data/test/default_setup/gold_predictions.json'))
      self.check_tf_events(CONFIG.model_dir, os.path.join(CONFIG.root_dir, 'data/test/default_setup/gold_train_tfevents'))
      self.check_tf_events(os.path.join(CONFIG.model_dir, 'eval_test'),
                           os.path.join(CONFIG.root_dir, 'data/test/default_setup/gold_eval_tfevents'))

  def check_detections(self, path_to_gold_detections):
    with open(os.path.join(CONFIG.model_dir, 'predictions/annotations_test.json')) as file:
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
        break

    summary = TrainNetCheckRegression.parse_tf_event_file(event_file)
    summary_gold = TrainNetCheckRegression.parse_tf_event_file(path_to_gold_tfevent)

    for tag_gold in summary_gold:
      if tag_gold == 'global_step/sec':  # Don't evaluate performance
        continue
      self.assertTrue(tag_gold in summary, msg='Tag: {0}'.format(tag_gold))
      self.assertEqual(len(summary[tag_gold]), len(summary_gold[tag_gold]), msg='Tag: {0}'.format(tag_gold))
      for val, val_gold in zip(summary[tag_gold], summary_gold[tag_gold]):
        self.assertAlmostEqual(val, val_gold, msg='Tag: {0}'.format(tag_gold))

