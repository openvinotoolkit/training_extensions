import tensorflow as tf
import tensorflow.contrib.slim as slim


@slim.add_arg_scope
def get_spatial_dims(tensor_or_shape, data_format='NHWC'):
  if isinstance(tensor_or_shape, (list, tuple)):
    input_shape = tensor_or_shape
  else:
    input_shape = tensor_or_shape.get_shape().as_list()

  assert data_format in ('NCHW', 'NHWC')
  assert len(input_shape) == 4

  if data_format == 'NHWC':
    height = input_shape[1]
    width = input_shape[2]
  else:
    height = input_shape[2]
    width = input_shape[3]
  return height, width


@slim.add_arg_scope
def channel_to_last(inputs, data_format='NHWC', scope=None):
  assert data_format in ('NCHW', 'NHWC')
  with tf.name_scope(scope, 'channel_to_last', [inputs]):
    return inputs if data_format == 'NHWC' else tf.transpose(inputs, perm=(0, 2, 3, 1))
