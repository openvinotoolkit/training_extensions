from __future__ import print_function
import math
import os
import pickle
import random

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.control_flow_ops import with_dependencies

from ssd_detector.networks.mobilenet_ssd import MobileNetSSD
from ssd_detector.readers.object_detector_json import ObjectDetectorJson
from ssd_detector.toolbox.loss import MultiboxLoss
from ssd_detector.toolbox.transformer import AnnotatedDataTransformer
from ssd_detector.toolbox.summary import create_tensors_and_streaming_ops_for_assigned_priors, \
  get_detailed_assigned_priors_summary_tf, write_histogram_2d_tf


def create_session(config, type):
  if type == 'train':
    random_seed = config.train.random_seed
  else:
    random_seed = 666

  np.random.seed(random_seed)
  tf.set_random_seed(random_seed)
  random.seed(random_seed)

  config_type = getattr(config, type).execution

  if hasattr(config_type, 'CUDA_VISIBLE_DEVICES'):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config_type.CUDA_VISIBLE_DEVICES

  intra_op_parallelism_threads = config_type.intra_op_parallelism_threads if \
    hasattr(config_type, 'intra_op_parallelism_threads') else 0
  inter_op_parallelism_threads = config_type.inter_op_parallelism_threads if \
    hasattr(config_type, 'inter_op_parallelism_threads') else 0
  session_config = tf.ConfigProto(allow_soft_placement=True,
                                  intra_op_parallelism_threads=intra_op_parallelism_threads,
                                  inter_op_parallelism_threads=inter_op_parallelism_threads)
  if hasattr(config_type, 'per_process_gpu_memory_fraction'):
    session_config.gpu_options.per_process_gpu_memory_fraction = config_type.per_process_gpu_memory_fraction
  if hasattr(config_type, 'allow_growth'):
    session_config.gpu_options.allow_growth = config_type.allow_growth

  return session_config


# pylint: disable=too-many-instance-attributes
class InputValData:
  # pylint: disable=too-many-arguments
  def __init__(self, batch_size, input_shape, json_path, classes, num_parallel_calls=2, prefetch_size=2):
    self.batch_size = batch_size
    self.input_shape = input_shape
    self.json_path = json_path
    self.num_parallel_calls = num_parallel_calls
    self.prefetch_size = prefetch_size

    ObjectDetectorJson.init_cache(self.json_path, cache_type='NONE', classes=classes)

    dataset, self.dataset_size = ObjectDetectorJson.create_dataset(self.json_path, classes=classes)
    _, self.transform_param = MobileNetSSD.create_transform_parameters(*input_shape[:2])
    self.transformer = AnnotatedDataTransformer(self.transform_param, is_training=False)

    print('Total evaluation steps: {}'.format(math.ceil(self.dataset_size / self.batch_size)))

    transform_fn = lambda value: ObjectDetectorJson.transform_fn(value, self.transformer)
    map_fn = lambda value: tf.py_func(transform_fn, [value], (tf.float32, tf.string))
    self.dataset = dataset.map(map_fn, num_parallel_calls=num_parallel_calls)
    self.dataset = self.dataset.batch(self.batch_size).prefetch(prefetch_size)

  def input_fn(self):
    images, annotation = self.dataset.make_one_shot_iterator().get_next()
    images.set_shape([None] + list(self.input_shape))
    return images, annotation

  @staticmethod
  def sample_data(json_path, num_samples, input_shape, classes, seed=666):
    if num_samples == 0:
      return None

    data, _ = ObjectDetectorJson.json_iterator(json_path, classes)
    data = [x for x in data()]
    # data = ObjectDetectorJson.convert_coco_to_toolbox_format(COCO(json_path), classes)

    ObjectDetectorJson.init_cache(json_path, cache_type='NONE', classes=classes)

    rng = random.Random(seed)
    selected_items = rng.sample(range(len(data)), num_samples)

    _, transform_param = MobileNetSSD.create_transform_parameters(*input_shape[:2])
    transformer = AnnotatedDataTransformer(transform_param, is_training=False)

    transform_fn = lambda value: ObjectDetectorJson.transform_fn(value, transformer, add_original_image=True)
    return [transform_fn(data[i]) for i in selected_items]


class InputTrainData:
  # pylint: disable=dangerous-default-value,too-many-arguments
  def __init__(self, batch_size, input_shape, json_path, cache_type='NONE', classes=['bg'],
               fill_with_current_image_mean=True, num_parallel_calls=4, prefetch_size=16):
    self.batch_size = batch_size
    self.input_shape = input_shape
    self.json_path = json_path
    self.cache_type = cache_type
    self.num_parallel_calls = num_parallel_calls
    self.prefetch_size = prefetch_size
    self.classes = classes

    ObjectDetectorJson.init_cache(self.json_path, cache_type, classes=classes)

    self.train_dataset, self.dataset_size = ObjectDetectorJson.create_dataset(self.json_path, classes)
    self.train_transform_param, _ = MobileNetSSD.create_transform_parameters(input_shape[0], input_shape[1],
                                                                             fill_with_current_image_mean)
    self.train_transformer = AnnotatedDataTransformer(self.train_transform_param, is_training=True)

  def input_fn(self):
    transform_fn = lambda value: ObjectDetectorJson.transform_fn(value, self.train_transformer,
                                                                 cache_type=self.cache_type)

    def transform_batch_fn(value):
      images = []
      annotations = []
      for val in value:
        img, annot = transform_fn(val)
        images.append(img)
        annotations.append(annot)
      return images, annotations

    map_fn_batch = lambda value: tf.py_func(transform_batch_fn, [value], (tf.float32, tf.string))

    dataset = self.train_dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=self.dataset_size))
    dataset = dataset.batch(self.batch_size).map(map_fn_batch, num_parallel_calls=self.num_parallel_calls)
    dataset = dataset.prefetch(self.prefetch_size)

    images, annotation = dataset.make_one_shot_iterator().get_next()

    images.set_shape([self.batch_size] + list(self.input_shape))
    return images, annotation


class InputInferData:
  # pylint: disable=too-many-arguments
  def __init__(self, path_to_video, input_shape, batch_size, num_parallel_calls=4, prefetch_size=32):
    self.path_to_video = path_to_video
    self.input_shape = input_shape
    self.cap = cv2.VideoCapture(self.path_to_video)
    self.num_parallel_calls = num_parallel_calls
    self.prefetch_size = prefetch_size
    self.batch_size = batch_size

  @staticmethod
  def transform_fn(value, transformer):
    image = pickle.loads(value).astype(np.float32)
    transformed_image, _ = transformer.transform(image, {})
    return transformed_image.astype(np.float32)


  def create_dataset(self):
    def generator():
      while self.cap.isOpened():
        ret, frame = self.cap.read()
        if ret:
          yield pickle.dumps(frame.astype(np.float32))

    return tf.data.Dataset.from_generator(generator, tf.string, tf.TensorShape([]))


  def input_fn(self):
    dataset = self.create_dataset()

    _, transform_param = MobileNetSSD.create_transform_parameters(*self.input_shape[:2])
    transformer = AnnotatedDataTransformer(transform_param, is_training=False)

    transform_fn = lambda value: InputInferData.transform_fn(value, transformer)
    map_fn = lambda value: tf.py_func(transform_fn, [value], tf.float32)
    dataset = dataset.map(map_fn, num_parallel_calls=self.num_parallel_calls).batch(self.batch_size).prefetch(
      self.prefetch_size)

    image = dataset.make_one_shot_iterator().get_next()
    image.set_shape([None] + list(self.input_shape))
    return image


# pylint: disable=too-many-locals,too-many-statements
def detection_model(features, labels, mode, params):
  num_classes = params['num_classes']
  initial_weights_path = params.get('initial_weights_path', '')
  log_dir = params['log_dir']
  collect_priors_summary = params['collect_priors_summary']

  data_format = params.get('data_format', 'NHWC')
  depth_multiplier = params.get('depth_multiplier', 1.0)
  priors_rule = params.get('priors_rule', 'caffe')
  custom_priors = params.get('priors', [])
  learning_rate = params.get('learning_rate', 0.01)
  steps_per_epoch = params.get('steps_per_epoch', 1)
  mobilenet_version = params.get('mobilenet_version', 'v2')
  weight_regularization = params.get('weight_regularization', 4e-5)
  optimizer_func = params.get('optimizer', lambda learning_rate: tf.train.AdagradOptimizer(learning_rate=learning_rate))

  # Override default FileWriter. Don't store the graph definition.
  # pylint: disable=protected-access
  tf.summary.FileWriterCache._cache[log_dir] = tf.summary.FileWriter(log_dir, graph=None)

  if callable(learning_rate):
    learning_rate = learning_rate()

  is_training = mode == tf.estimator.ModeKeys.TRAIN

  ssd = MobileNetSSD(input_tensor=features, num_classes=num_classes, depth_multiplier=depth_multiplier,
                     is_training=is_training, data_format=data_format, priors_rule=priors_rule,
                     priors=custom_priors, mobilenet_version=mobilenet_version,
                     weight_regularization=weight_regularization)  # 1. Build model

  if mode == tf.estimator.ModeKeys.PREDICT:
    decoded_predictions = ssd.detection_output(use_plain_caffe_format=False)
    return tf.estimator.EstimatorSpec(mode, predictions=decoded_predictions)

  assert mode in(tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL)
  targets = ssd.create_targets(labels)  # 2. Build GT from annotation

  if collect_priors_summary:
    with tf.name_scope('summary/'):
      assigned_priors = create_tensors_and_streaming_ops_for_assigned_priors(targets, ssd.priors_info, num_classes)
      detailed_assigned_priors = get_detailed_assigned_priors_summary_tf(assigned_priors, ssd.priors_info)

  loss_func = MultiboxLoss(neg_pos_ratio=3.0)  # 3. Build loss-object

  eval_iteration = tf.get_variable('eval_iteration', initializer=0, dtype=tf.int32, trainable=False)
  if mode == tf.estimator.ModeKeys.EVAL:
    eval_print_steps = steps_per_epoch // 50
    eval_print_steps = 1 if eval_print_steps == 0 else eval_print_steps

    every_eval_print_steps = tf.equal(tf.mod(eval_iteration + 1, eval_print_steps), 0)
    eval_iteration = tf.assign(eval_iteration, eval_iteration + 1)
    targets = with_dependencies([eval_iteration], targets)

    loss = loss_func.eval_summary(targets, ssd.predictions)
    loss = tf.cond(every_eval_print_steps,
                   lambda: tf.Print(loss, [tf.round(100 * eval_iteration / steps_per_epoch), loss], '[%][loss]: '),
                   lambda: loss)

    eval_metric_ops = {}
    for key, val in loss_func.eval_tensors.items():
      eval_metric_ops['loss_function/' + key] = tf.metrics.mean(val)

    if collect_priors_summary:
      for key, metric_ops in assigned_priors.items():  # We need only update ops
        eval_metric_ops[key] = metric_ops

      for key, assigned_priors_tensor in detailed_assigned_priors.items():
        eval_metric_ops['prior_histogram/' + key] = (assigned_priors_tensor, tf.no_op())

    decoded_predictions = ssd.detection_output(use_plain_caffe_format=False)
    eval_metric_ops['predictions'] = tf.contrib.metrics.streaming_concat(decoded_predictions)

    return tf.estimator.EstimatorSpec(
      mode,
      loss=loss,
      eval_metric_ops=eval_metric_ops
    )

  assert mode == tf.estimator.ModeKeys.TRAIN
  if initial_weights_path:
    ssd.load_weights(initial_weights_path)

  bboxes = ssd._decode_boxes(ssd.predictions['locs'], priors=ssd.priors[0, 0], variance=ssd.priors[0, 1])
  loss = loss_func.loss(targets, ssd.predictions, bboxes)  # 4. Compute loss with NMS

  if collect_priors_summary:
    with tf.name_scope('summary/'):
      loss = with_dependencies([operation for key, (_, operation) in assigned_priors.items()], loss)

    for name, assigned_priors_tensor in detailed_assigned_priors.items():
      tf.summary.scalar(name, tf.reduce_sum(assigned_priors_tensor))

    py_func_ops = []
    priors_dir = os.path.join(log_dir, 'priors')

    with tf.name_scope('write_histogram'):
      every_epoch = tf.equal(tf.mod(tf.train.get_global_step() + 1, steps_per_epoch), 0)
      for name, (group, _) in assigned_priors.items():
        def write_hist2d():
          # pylint: disable=cell-var-from-loop
          return tf.py_func(write_histogram_2d_tf,
                            [group, pickle.dumps(ssd.priors_info), name, tf.train.get_global_step(), priors_dir],
                            tf.bool)

        write_hist2d_once_per_epoch = tf.cond(every_epoch, write_hist2d, tf.no_op)
        py_func_ops.append(write_hist2d_once_per_epoch)

      loss = with_dependencies(py_func_ops, loss)

  optimizer = optimizer_func(learning_rate)
  tf.summary.scalar('learning_rate', learning_rate)

  regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  regularization_loss = tf.add_n(regularization_losses, name='loss_function/regularization_losses_sum')
  total_loss = tf.add(loss, regularization_loss, name='loss_function/total_loss')

  tf.summary.scalar('loss_function/regularization_loss', regularization_loss)

  with tf.variable_scope('train_loop'):
    train_op = optimizer.minimize(total_loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)
