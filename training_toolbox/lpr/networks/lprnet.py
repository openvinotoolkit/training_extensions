import tensorflow as tf
import tensorflow.contrib.slim as slim

class LPRNet():
  # Function for generation characters range

  # Fire block
  @staticmethod
  def fire_block(input, outputs):
    fire = slim.conv2d(input, outputs / 4, [1, 1])
    fire = slim.conv2d(fire, outputs / 4, [3, 3])
    fire = slim.conv2d(fire, outputs, [1, 1])
    return fire

  # Small Fire block
  @staticmethod
  def small_fire_block(input, outputs):
    fire = slim.conv2d(input, outputs / 4, [1, 1])
    fire = slim.conv2d(fire, outputs / 4, [3, 1])
    fire = slim.conv2d(fire, outputs / 4, [1, 3])
    fire = slim.conv2d(fire, outputs, [1, 1])
    return fire

  # Inception-ResNet like block
  @staticmethod
  def resinc_block(input, outputs):
    inputs = int(input.get_shape()[3])
    if inputs == outputs:
      res = input
    else:
      res = slim.conv2d(input, outputs, [1, 1])
    inc1 = slim.conv2d(input, outputs / 8, [1, 1])
    inc2 = slim.conv2d(input, outputs / 8, [1, 1])
    inc2 = slim.conv2d(inc2, outputs / 8, [3, 1])
    inc2 = slim.conv2d(inc2, outputs / 8, [1, 3])
    conc = tf.concat(3, [inc1, inc2])
    inc = slim.conv2d(conc, outputs, [1, 1])
    return res + inc

  # basic_block = fire_block
  basic_block = small_fire_block

  # basic_block = resinc_block

  # Convolution block for CNN
  @staticmethod
  def convolution_block(input, outputs, stride, **kwargs):
    scope = kwargs.pop('scope', None)
    # cr = slim.conv2d(input, outputs, [3, 3], scope=scope)
    cr = LPRNet.basic_block(input, outputs)
    mp = slim.max_pool2d(cr, [3, 3], stride=(stride, 1), padding='VALID', scope=scope)
    return mp

  @staticmethod
  def enet_input_block(input, **kwargs):
    scope = kwargs.pop('scope', None)
    input1 = slim.conv2d(input, 61, [3, 3], stride=(2, 1), padding='VALID', scope=scope)
    input2 = slim.avg_pool2d(input, [3, 3], stride=(2, 1), padding='VALID', scope=scope)
    step1 = tf.concat(3, [input1, input2])
    step2 = LPRNet.basic_block(step1, 128)
    step2 = slim.max_pool2d(step2, [2, 2], stride=(1, 1), padding='VALID', scope=scope)
    return step2

  @staticmethod
  def std_input_block(input):
    return slim.stack(input, LPRNet.convolution_block, [(64, 1), (128, 2)])

  @staticmethod
  def mixed_input_block(input):
    cnn = slim.conv2d(input, 64, [3, 3])
    cnn = slim.max_pool2d(cnn, [3, 3], stride=(1, 1), padding='VALID')
    cnn = LPRNet.basic_block(cnn, 128)
    cnn = slim.max_pool2d(cnn, [3, 3], stride=(2, 1), padding='VALID')
    return cnn

  input_block = mixed_input_block

  @staticmethod
  def lprnet(input):
    with slim.arg_scope([slim.fully_connected, slim.conv2d], activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm, weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
      cnn = LPRNet.input_block(input)
      cnn = LPRNet.basic_block(cnn, 256)
      cnn = LPRNet.convolution_block(cnn, 256, 2)

      cnn = slim.dropout(cnn)
      cnn = slim.conv2d(cnn, 256, [4, 1], padding='VALID')
      cnn = slim.dropout(cnn)

      return cnn

