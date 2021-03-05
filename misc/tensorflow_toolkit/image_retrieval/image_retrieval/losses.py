"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import math

import tensorflow as tf
import tensorflow.keras.backend as K


# pylint: disable=abstract-method
class AMSoftmaxLogits(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(AMSoftmaxLogits, self).__init__(**kwargs)
        self.units = units
        self.kernel = None

    def build(self, input_shape):
        # pylint: disable=no-value-for-parameter
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer=tf.keras.initializers.get('glorot_uniform'))

    # pylint: disable=arguments-differ
    def call(self, inputs):
        kernel = tf.nn.l2_normalize(self.kernel, axis=0, epsilon=1e-13)
        inputs = tf.nn.l2_normalize(inputs, axis=1, epsilon=1e-13)

        cos_theta = K.dot(inputs, kernel)
        cos_theta = K.clip(cos_theta, -1.0, 1.0)

        return tf.identity(cos_theta)


def am_softmax_loss(num_classes, s, m):
    if s is None:
        s = math.sqrt(2) * math.log(num_classes - 1)

    def loss(y_true, cos_theta):
        phi_theta = cos_theta - m
        y_true_reshaped = tf.reshape(y_true, (-1,))
        y_true_one_hot = tf.one_hot(tf.cast(y_true_reshaped, tf.int32), num_classes)

        output = tf.where(tf.cast(y_true_one_hot, tf.bool), phi_theta, cos_theta)
        output = output * s

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true, output)

        return loss

    return loss


def triplet_loss(margin):
    def loss(labels, embeddings):
        from tensorflow_addons.losses import triplet_semihard_loss
        return triplet_semihard_loss(labels, embeddings, margin=margin)

    return loss
