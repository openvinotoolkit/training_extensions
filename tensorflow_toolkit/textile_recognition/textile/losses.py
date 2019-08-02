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

import tensorflow as tf
import tensorflow.keras.backend as K


class AMSoftmax(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(AMSoftmax, self).__init__(**kwargs)
        self.units = units
        self.kernel = None

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer=tf.keras.initializers.get('glorot_uniform'))

    def call(self, inputs, **kwargs):
        kernel = tf.nn.l2_normalize(self.kernel, axis=0, epsilon=1e-13)

        # tf.print('kernel.shape', kernel.shape)
        # tf.print('inputs.shape', inputs.shape)

        # tf.print('kernel.norm', tf.norm(kernel, axis=0))
        # tf.print('inputs.norm', tf.norm(inputs, axis=1))

        cos_theta = K.dot(inputs, kernel)

        cos_theta = K.clip(cos_theta, -1.0, 1.0)
        #
        # phi_theta = cos_theta - self.m




        # e_cos_theta = K.exp(self.s * cos_theta)
        #
        # e_psi = K.exp(self.s * phi_theta)
        #
        # sum_x = K.sum(e_cos_theta, axis=-1, keepdims=True)
        #
        # temp = e_psi - e_cos_theta
        #
        # temp = temp + sum_x
        #
        # output = e_psi / temp

        return tf.identity(cos_theta)


def am_softmax_loss(num_classes, s, m, alpha=1.0, gamma=0.0):
    def amsoftmax_loss(y_true, cos_theta):
        phi_theta = cos_theta - m

        y_true = tf.reshape(y_true, (-1,))
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)


        output = tf.where(tf.cast(y_true_one_hot, tf.bool), phi_theta, cos_theta)

        output = output * s


        # d1 = K.sum(y_true * y_pred, axis=-1)
        #
        # d1 = K.clip(d1, K.epsilon(), 1.0 - K.epsilon())
        #
        # d1 = alpha * (tf.math.pow(1.0 - d1, gamma)) * K.log(d1)

        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(y_true_one_hot, output)
        return loss

    def loss(labels, embeddings):
        cos_theta = AMSoftmax(num_classes)(embeddings)
        return amsoftmax_loss(labels, cos_theta)

    return loss


def triplet_loss(margin):

    def loss(labels, embeddings):
        from tensorflow_addons.losses import triplet_semihard_loss
        return triplet_semihard_loss(labels, embeddings, margin=margin)

    return loss
