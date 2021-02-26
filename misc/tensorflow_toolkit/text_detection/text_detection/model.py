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

""" This module contains architecture of text detector. """

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, SeparableConv2D


def mobilenet_v2(inputs, original_stride, weights_decay=0):
    """ Contains MobileNet_v2 definition.

    This is NOT original MobileNet_v2.
    * Conv2D biases are ON
    * Extra 1x1 convs are added (SeparableConv2D instead of DepthwiseConv2D)
    * First mobile_net_block contains more layers than original MobileNet_v2.

    """

    def mobile_net_block(inputs, expand_to, strided, num_outputs):
        # Following expand layer should be only if input_filters < output_fildetes
        net = Conv2D(filters=expand_to, kernel_size=1, padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(weights_decay))(inputs)
        net = BatchNormalization()(net)
        net = ReLU(max_value=6)(net)

        net = SeparableConv2D(filters=expand_to, kernel_size=3, strides=2 if strided else 1,
                              padding='same',
                              kernel_regularizer=tf.keras.regularizers.l2(weights_decay))(net)
        net = BatchNormalization()(net)
        net = ReLU(max_value=6)(net)

        net = Conv2D(filters=num_outputs, kernel_size=1, padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(weights_decay))(net)
        net = BatchNormalization()(net)

        if not strided and net.get_shape().as_list()[-1] == inputs.get_shape().as_list()[-1]:
            return tf.keras.layers.Add()([inputs, net])

        return net

    end_points = {}

    net = BatchNormalization(name='data_bn')(inputs)

    net = Conv2D(filters=32, kernel_size=3, strides=2, padding='same',
                 kernel_regularizer=tf.keras.regularizers.l2(weights_decay))(net)
    net = BatchNormalization()(net)
    net = ReLU(max_value=6)(net)

    net = mobile_net_block(net, strided=False, expand_to=32, num_outputs=16)
    end_points['2x'] = net

    net = mobile_net_block(net, strided=True, expand_to=96, num_outputs=24)
    net = mobile_net_block(net, strided=False, expand_to=144, num_outputs=24)
    end_points['4x'] = net

    net = mobile_net_block(net, strided=True, expand_to=144, num_outputs=32)

    net = mobile_net_block(net, strided=False, expand_to=192, num_outputs=32)
    net = mobile_net_block(net, strided=False, expand_to=192, num_outputs=32)
    if original_stride:
        end_points['8x'] = net

    net = mobile_net_block(net, strided=original_stride, expand_to=192, num_outputs=64)
    net = mobile_net_block(net, strided=False, expand_to=384, num_outputs=64)
    net = mobile_net_block(net, strided=False, expand_to=384, num_outputs=64)
    net = mobile_net_block(net, strided=False, expand_to=384, num_outputs=64)
    if not original_stride:
        end_points['8x'] = net

    net = mobile_net_block(net, strided=not original_stride, expand_to=384, num_outputs=96)
    net = mobile_net_block(net, strided=False, expand_to=576, num_outputs=96)
    net = mobile_net_block(net, strided=False, expand_to=576, num_outputs=96)
    end_points['16x'] = net

    net = mobile_net_block(net, strided=True, expand_to=576, num_outputs=160)

    net = mobile_net_block(net, strided=False, expand_to=960, num_outputs=160)
    net = mobile_net_block(net, strided=False, expand_to=960, num_outputs=160)
    net = mobile_net_block(net, strided=False, expand_to=960, num_outputs=320)

    net = Conv2D(filters=1280, kernel_size=1, padding='same',
                 kernel_regularizer=tf.keras.regularizers.l2(weights_decay))(net)
    net = BatchNormalization()(net)
    net = ReLU(max_value=6)(net)
    end_points['32x'] = net

    return end_points


def fcn_head(inputs, num_classes, name, weights_decay=0):
    """ Defines FCN head. """

    x32 = tf.keras.layers.Conv2D(
        filters=num_classes, strides=1, kernel_size=1,
        kernel_regularizer=tf.keras.regularizers.l2(weights_decay))(inputs['32x'])

    x32_upscaled = tf.keras.layers.UpSampling2D(interpolation='bilinear')(x32)
    x16 = tf.keras.layers.Add()([tf.keras.layers.Conv2D(
        filters=num_classes, strides=1, kernel_size=1,
        kernel_regularizer=tf.keras.regularizers.l2(weights_decay))(inputs['16x']), x32_upscaled])

    x16_upscaled = tf.keras.layers.UpSampling2D(interpolation='bilinear')(x16)
    x08 = tf.keras.layers.Add()([tf.keras.layers.Conv2D(
        filters=num_classes, strides=1, kernel_size=1,
        kernel_regularizer=tf.keras.regularizers.l2(weights_decay))(inputs['8x']), x16_upscaled])

    x08_upscaled = tf.keras.layers.UpSampling2D(interpolation='bilinear')(x08)
    x04 = tf.keras.layers.Add(name=name)([
        tf.keras.layers.Conv2D(
            filters=num_classes, strides=1, kernel_size=1,
            kernel_regularizer=tf.keras.regularizers.l2(weights_decay))(inputs['4x']), x08_upscaled
    ])

    return x04


def keras_applications_mobilenetv2(inputs, alpha):
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

    base_model = MobileNetV2(alpha=alpha, include_top=False,
                             weights='imagenet', input_tensor=inputs)

    outputs = {'4x': base_model.get_layer('block_2_add').output,
               '8x': base_model.get_layer('block_5_add').output,
               '16x': base_model.get_layer('block_12_add').output,
               '32x': base_model.get_layer('out_relu').output}

    return outputs


def keras_applications_vgg16(inputs):
    from tensorflow.keras.applications.vgg16 import VGG16

    base_model = VGG16(input_tensor=inputs, weights='imagenet', include_top=False)

    outputs = {'4x': base_model.get_layer('block3_conv3').output,
               '8x': base_model.get_layer('block4_conv3').output,
               '16x': base_model.get_layer('block5_conv3').output,
               '32x': base_model.get_layer('block5_pool').output}

    return outputs


def keras_applications_resnet50(inputs):
    from tensorflow.keras.applications.resnet50 import ResNet50

    base_model = ResNet50(input_tensor=inputs, weights='imagenet', include_top=False)

    try:
        outputs = {'4x': base_model.get_layer('activation_9').output,
                   '8x': base_model.get_layer('activation_21').output,
                   '16x': base_model.get_layer('activation_39').output,
                   '32x': base_model.get_layer('activation_48').output}
    except:
        outputs = {'4x': base_model.get_layer('activation_58').output,
                   '8x': base_model.get_layer('activation_70').output,
                   '16x': base_model.get_layer('activation_88').output,
                   '32x': base_model.get_layer('activation_97').output}

    return outputs


def keras_applications_xception(inputs):
    from tensorflow.keras.applications.xception import Xception

    base_model = Xception(input_tensor=inputs, weights='imagenet', include_top=False)

    tf.keras.utils.plot_model(base_model, 'model.png')

    for layer in base_model.layers:
        print(layer.name, layer.output)

    outputs = {
        '4x': tf.keras.layers.ZeroPadding2D(((0, 1), (0, 1)))(base_model.get_layer('add').output),
        '8x': base_model.get_layer('add_1').output,
        '16x': base_model.get_layer('add_10').output,
        '32x': base_model.get_layer('block14_sepconv2_act').output}

    return outputs


def pixel_link_model(inputs, config):
    """ PixelLink architecture. """

    if config['model_type'] == 'mobilenet_v2_ext':
        backbone = mobilenet_v2(inputs, original_stride=False,
                                weights_decay=config['weights_decay'])
    elif config['model_type'] == 'ka_resnet50':
        backbone = keras_applications_resnet50(inputs)
    elif config['model_type'] == 'ka_vgg16':
        backbone = keras_applications_vgg16(inputs)
    elif config['model_type'] == 'ka_mobilenet_v2_1_0':
        backbone = keras_applications_mobilenetv2(inputs, alpha=1.0)
    elif config['model_type'] == 'ka_mobilenet_v2_1_4':
        backbone = keras_applications_mobilenetv2(inputs, alpha=1.4)
    elif config['model_type'] == 'ka_xception':
        backbone = keras_applications_xception(inputs)

    segm_logits = fcn_head(backbone, num_classes=2, name='segm_logits',
                           weights_decay=config['weights_decay'])
    link_logits = fcn_head(backbone, num_classes=16, name='link_logits_',
                           weights_decay=config['weights_decay'])

    new_shape = tf.shape(link_logits)[1], tf.shape(link_logits)[2], 8, 2
    link_logits = tf.keras.layers.Reshape(new_shape, name='link_logits')(link_logits)

    return tf.keras.Model(inputs, [segm_logits, link_logits])
