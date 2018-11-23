"""
  MobileNet + SSD.
"""

import math

import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python import pywrap_tensorflow
from slim.nets.mobilenet_v1 import mobilenet_v1_base
from slim.nets.mobilenet.mobilenet_v2 import mobilenet_base as mobilenet_v2_base
from slim.nets.mobilenet import mobilenet, mobilenet_v2

from ssd_detector.toolbox.ssd_base import SSDBase
from ssd_detector.toolbox.transformer import ResizeParameter, ExpansionParameter, TransformationParameter, DistortionParameter


class MobileNetSSD(SSDBase):
  def __init__(self, num_classes, input_tensor, is_training=False, data_format='NHWC',
               priors_rule='object_detection_api', priors=[],
               mobilenet_version='v2', depth_multiplier=1.0, min_depth=16,
               weight_regularization=4e-5):
    """

    Args:
      num_classes: Number of classes including a background class.
      input_tensor: Input 4D tensor.
      is_training: Is training or inference stage.
      data_format: 'NHWC' or 'NCHW'.
      priors_rule: 'caffe', 'object_detection_api', 'custom'.
      priors: List of list of prior sizes (relative sizes). Only for priors_rule='custom'.
      mobilenet_version: 'v1' or 'v2'.
      depth_multiplier: MobileNet depth multiplier.
      min_depth: Minimum channels count in MobileNet.
      weight_regularization: l2 weight regularization scale.
    """
    assert data_format in ['NHWC', 'NCHW']
    assert priors_rule in ['caffe', 'object_detection_api', 'custom']

    self.data_format = data_format
    if self.data_format == 'NCHW':
      input_tensor = tf.transpose(input_tensor, [0, 3, 1, 2])
    self.input_shape = input_tensor.get_shape().as_list()
    self.input_tensor = input_tensor

    if self.data_format == 'NCHW':
      spatial_dim_axis = [2, 3]
    elif self.data_format == 'NHWC':
      spatial_dim_axis = [1, 2]

    self.version = mobilenet_version

    super(MobileNetSSD, self).__init__(num_classes=num_classes, input_shape=self.input_shape, data_format=data_format)

    self.is_training = is_training

    if mobilenet_version == 'v2':
      mobilenet_base = mobilenet_v2_base
      base_scope = mobilenet_v2.training_scope
      base_layers = ['layer_7/output', 'layer_15/expansion_output', 'layer_19']
    elif mobilenet_version == 'v1':
      mobilenet_base = mobilenet_v1_base
      base_scope = mobilenet.training_scope
      base_layers = ['Conv2d_5_pointwise', 'Conv2d_11_pointwise', 'Conv2d_13_pointwise']
    else:
      tf.logging.error('Wrong MobileNet version = {}'.format(mobilenet_version))
      exit(0)

    def scope_fn():
      batch_norm_params = {
        'is_training': self.is_training,
        'center': True,
        'scale': True,
        'decay': 0.9997,
        'epsilon': 0.001,
        'fused': True,
        'data_format': data_format
      }
      affected_ops = [slim.conv2d, slim.separable_conv2d, slim.conv2d_transpose]
      with (slim.arg_scope([slim.batch_norm], **batch_norm_params)):
        with slim.arg_scope(
            affected_ops,
            weights_regularizer=slim.l2_regularizer(scale=float(weight_regularization)),
            weights_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.03),
            activation_fn=tf.nn.relu6,
            normalizer_fn=slim.batch_norm) as sc:
          return sc

    with slim.arg_scope(base_scope(is_training=None)):
      with slim.arg_scope(scope_fn()):
        _, image_features = mobilenet_base(
          self.input_tensor,
          final_endpoint=base_layers[-1],
          depth_multiplier=depth_multiplier,
          min_depth=min_depth,
          use_explicit_padding=False,
          is_training=self.is_training)

    head_feature_map_names = base_layers[-2:]
    head_feature_map_tensors = [image_features[name] for name in head_feature_map_names]

    feature_map = image_features[base_layers[-1]]

    depths = [512, 256, 256, 128]
    depths = [int(d * depth_multiplier) for d in depths]
    with tf.variable_scope('extra_features'):
        with slim.arg_scope(scope_fn()):
          for i, depth in enumerate(depths):
            intermediate_layer = slim.conv2d(feature_map, int(depth / 2), [1, 1], stride=1,
                                             scope='intermediate_{0}'.format(i + 1))
            # feature_map = slim.conv2d(intermediate_layer, depth, [3, 3], stride=2, scope='feature_map_{0}'.format(i + 1))
            feature_map = slim.separable_conv2d(
              intermediate_layer,
              None, [3, 3],
              depth_multiplier=1,
              padding='SAME',
              stride=2,
              scope='feature_map_dw_{0}'.format(i + 1))

            output_feature_name = 'feature_map_{0}'.format(i + 1)
            feature_map = slim.conv2d(
              feature_map,
              int(depth), [1, 1],
              padding='SAME',
              stride=1,
              scope=output_feature_name)

            head_feature_map_names.append(output_feature_name)
            head_feature_map_tensors.append(feature_map)

    variances = [0.1, 0.1, 0.2, 0.2]

    if priors_rule == 'caffe':
      scale = [0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
      dicts = self._create_caffe_priors(self.input_shape, spatial_dim_axis, scale, variances,
                                        head_feature_map_tensors, head_feature_map_names)
    elif priors_rule == 'object_detection_api':
      scale = [0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.]
      dicts = self._create_obj_det_priors(self.input_shape, spatial_dim_axis, scale, variances,
                                          head_feature_map_tensors, head_feature_map_names)
    elif priors_rule == 'custom':
      assert len(priors) == len(head_feature_map_tensors)
      dicts = self._create_custom_priors(self.input_shape, spatial_dim_axis, priors, variances,
                                         head_feature_map_tensors, head_feature_map_names)

    with slim.arg_scope(scope_fn()):
      self.create_heads(head_feature_map_tensors, dicts)


  @staticmethod
  def _get_spatial_step(input_shape, tensor, spatial_dim_axis):
    output_shape = tensor.get_shape().as_list()
    return [float(input_shape[axis]) / float(output_shape[axis]) for axis in spatial_dim_axis]

  @staticmethod
  def _create_caffe_priors(input_shape, spatial_dim_axis, scale, variances, tensors, names):
    width, _ = [input_shape[axis] for axis in spatial_dim_axis]
    dicts = []
    for i, scale_ in enumerate(scale):
      dict_ = dict(prefix=names[i], step=MobileNetSSD._get_spatial_step(input_shape, tensors[i], spatial_dim_axis),
                   variance=variances, clip=True)
      if i == 0:
        dict_.update(dict(min_sizes=[width * 0.1], max_sizes=[width * scale_], aspect_ratios=[2]))
      else:
        dict_.update(dict(min_sizes=[width * scale_], aspect_ratios=[2, 3]))
      dicts.append(dict_)

    return dicts

  @staticmethod
  def _create_obj_det_priors(input_shape, spatial_dim_axis, scale, variances, tensors, names):
    width, _ = [input_shape[axis] for axis in spatial_dim_axis]

    aspect_ratio = [2., 0.5, 3., 1. / 3.]
    box_specs = []

    for i in range(len(scale) - 1):
      specs = []
      if i == 0:
        specs.append([scale[i] * 0.5, 1.])
        specs.append([scale[i], 2.])
        specs.append([scale[i], 0.5])
      else:
        specs.append([scale[i], 1.])
        specs.append([math.sqrt(scale[i] * scale[i + 1]), 1.])
        for asp_rat in aspect_ratio:
          specs.append([scale[i], asp_rat])

      specs = [[width * sc, ar] for sc, ar in specs]
      box_specs.append(specs)

    dicts = [dict(box_specs=box_specs[i], prefix=names[i], variance=variances, clip=True,
                  step=MobileNetSSD._get_spatial_step(input_shape, tensors[i], spatial_dim_axis))
             for i in range(len(scale) - 1)]

    return dicts

  @staticmethod
  def _create_custom_priors(input_shape, spatial_dim_axis, priors, variances, tensors, names):
    dicts = []
    for (i, sizes) in enumerate(priors):
      dict_ = dict(prefix=names[i], step=MobileNetSSD._get_spatial_step(input_shape, tensors[i], spatial_dim_axis),
                   variance=variances, clustered_sizes=sizes, clip=True)
      dicts.append(dict_)

    return dicts

  @staticmethod
  def create_transform_parameters(height, width, fill_with_current_image_mean=True):
    scale = 2.0 / 255.0
    mean_value = 255.0 / 2.0
    resize_param = ResizeParameter(height=height, width=width)
    expand_param = ExpansionParameter(prob=0.5, max_expand_ratio=1.5,
                                      fill_with_current_image_mean=fill_with_current_image_mean)
    distort_param = DistortionParameter(brightness_prob=0.5, brightness_delta=32., contrast_prob=0.5,
                                        contrast_lower=0.5, contrast_upper=1.5,
                                        hue_prob=0.5, hue_delta=18, saturation_prob=0.5, saturation_lower=0.5,
                                        saturation_upper=1.5, random_order_prob=0.5)
    train_param = TransformationParameter(mirror=True, resize_param=resize_param,
                                          expand_param=expand_param,
                                          scale=scale, mean_value=mean_value,
                                          noise_param=None, distort_param=distort_param)

    val_resize_param = ResizeParameter(height=height, width=width, interp_mode=[cv2.INTER_LINEAR])
    val_param = TransformationParameter(resize_param=val_resize_param, scale=scale, mean_value=mean_value)
    return train_param, val_param

  def load_weights(self, weights_path):
    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    reader = pywrap_tensorflow.NewCheckpointReader(weights_path)
    ckp_vars_to_shape = reader.get_variable_to_shape_map()
    if 'global_step' in ckp_vars_to_shape:
      del ckp_vars_to_shape['global_step']  # Do not restore global_step

    def __get_vars_to_restore(model_vars, ckp_vars_to_shape, prefixes):
      vars_to_restore = dict()
      skip_vars = []

      stat = dict()
      for prefix in prefixes:
        stat[prefix] = 0

      for var in model_vars:
        var_name = var.name[:-2]
        var_shape = var.shape.as_list()

        skip_var = True
        for prefix in prefixes:
          if prefix + var_name in ckp_vars_to_shape and ckp_vars_to_shape[prefix + var_name] == var_shape:
            vars_to_restore[prefix + var_name] = var_name
            stat[prefix] += 1
            skip_var = False
            break

        if skip_var:
          skip_vars.append(var)

      return vars_to_restore, skip_vars, stat

    prefixes = ['',                   # For classification models and checkpoints
                'FeatureExtractor/']  # For models from object detection API

    vars_to_restore, skip_vars, stat = __get_vars_to_restore(variables, ckp_vars_to_shape, prefixes)

    for prefix, restored_vars in stat.items():
      tf.logging.info("For the prefix '{0}' were found {1} weights".format(prefix, restored_vars))

    try:
      with tf.name_scope('load_weights'):
        tf.train.init_from_checkpoint(weights_path, vars_to_restore)
      tf.logging.info("Values were loaded for {} tensors!".format(len(vars_to_restore.keys())))
      tf.logging.info("Values were not loaded for {} tensors:".format(len(skip_vars)))
      for var in skip_vars:
        tf.logging.info("  {}".format(var))
    except ValueError as e:
      tf.logging.error("Weights was not loaded at all!")
      tf.logging.error(e)
      exit(1)
