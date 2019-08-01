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

""" This module contains architecture of Text Recognition model."""

import tensorflow as tf
from tensorflow.contrib import rnn
import tensorflow.contrib.slim as slim


class TextRecognition:
    """ Text recognition model definition. """

    def __init__(self, is_training, num_classes, backbone_dropout=0.0):
        self.is_training = is_training
        self.lstm_dim = 256
        self.num_classes = num_classes
        self.backbone_dropout = backbone_dropout

    def __call__(self, inputdata):
        with tf.variable_scope('shadow'):
            features = self.feature_extractor(inputdata=inputdata)
            logits = self.encoder_decoder(inputdata=tf.squeeze(features, axis=1))

        return logits

    # pylint: disable=too-many-locals
    def feature_extractor(self, inputdata):
        """ Extracts features from input text image. """

        with slim.arg_scope([slim.conv2d], padding='SAME',
                            weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.00025),
                            biases_initializer=None, activation_fn=None):
            with slim.arg_scope([slim.batch_norm], updates_collections=None):
                bn0 = slim.batch_norm(inputdata, 0.9, scale=True, is_training=self.is_training,
                                      activation_fn=None)

                dropout1 = slim.dropout(bn0, keep_prob=1.0 - self.backbone_dropout,
                                        is_training=self.is_training)
                conv1 = slim.conv2d(dropout1, num_outputs=64, kernel_size=3)
                bn1 = slim.batch_norm(conv1, 0.9, scale=True, is_training=self.is_training,
                                      activation_fn=tf.nn.relu)
                pool1 = slim.max_pool2d(bn1, kernel_size=2, stride=2)

                dropout2 = slim.dropout(pool1, keep_prob=1.0 - self.backbone_dropout,
                                        is_training=self.is_training)
                conv2 = slim.conv2d(dropout2, num_outputs=128, kernel_size=3)
                bn2 = slim.batch_norm(conv2, 0.9, scale=True, is_training=self.is_training,
                                      activation_fn=tf.nn.relu)
                pool2 = slim.max_pool2d(bn2, kernel_size=2, stride=2)

                dropout3 = slim.dropout(pool2, keep_prob=1.0 - self.backbone_dropout,
                                        is_training=self.is_training)
                conv3 = slim.conv2d(dropout3, num_outputs=256, kernel_size=3)
                bn3 = slim.batch_norm(conv3, 0.9, scale=True, is_training=self.is_training,
                                      activation_fn=tf.nn.relu)

                dropout4 = slim.dropout(bn3, keep_prob=1.0 - self.backbone_dropout,
                                        is_training=self.is_training)
                conv4 = slim.conv2d(dropout4, num_outputs=256, kernel_size=3)
                bn4 = slim.batch_norm(conv4, 0.9, scale=True, is_training=self.is_training,
                                      activation_fn=tf.nn.relu)
                pool4 = slim.max_pool2d(bn4, kernel_size=[2, 1], stride=[2, 1])

                dropout5 = slim.dropout(pool4, keep_prob=1.0 - self.backbone_dropout,
                                        is_training=self.is_training)
                conv5 = slim.conv2d(dropout5, num_outputs=512, kernel_size=3)
                bn5 = slim.batch_norm(conv5, 0.9, scale=True, is_training=self.is_training,
                                      activation_fn=tf.nn.relu)

                dropout6 = slim.dropout(bn5, keep_prob=1.0 - self.backbone_dropout,
                                        is_training=self.is_training)
                conv6 = slim.conv2d(dropout6, num_outputs=512, kernel_size=3)
                bn6 = slim.batch_norm(conv6, 0.9, scale=True, is_training=self.is_training,
                                      activation_fn=tf.nn.relu)
                pool6 = slim.max_pool2d(bn6, kernel_size=[2, 1], stride=[2, 1])

                dropout7 = slim.dropout(pool6, keep_prob=1.0 - self.backbone_dropout,
                                        is_training=self.is_training)
                conv7 = slim.conv2d(dropout7, num_outputs=512, kernel_size=2, stride=[2, 1])
                bn7 = slim.batch_norm(conv7, 0.9, scale=True, is_training=self.is_training,
                                      activation_fn=tf.nn.relu)

        return bn7

    def encoder_decoder(self, inputdata):
        """ LSTM-based encoder-decoder module. """

        with tf.variable_scope('LSTMLayers'):
            [batch_size, width, _] = inputdata.get_shape().as_list()

            with tf.variable_scope('encoder'):
                forward_cells = []
                backward_cells = []

                for _ in range(2):
                    forward_cells.append(tf.nn.rnn_cell.LSTMCell(self.lstm_dim))
                    backward_cells.append(tf.nn.rnn_cell.LSTMCell(self.lstm_dim))

                encoder_layer, _, _ = rnn.stack_bidirectional_dynamic_rnn(
                    forward_cells, backward_cells, inputdata, dtype=tf.float32)

            with tf.variable_scope('decoder'):
                forward_cells = []
                backward_cells = []

                for _ in range(2):
                    forward_cells.append(tf.nn.rnn_cell.LSTMCell(self.lstm_dim))
                    backward_cells.append(tf.nn.rnn_cell.LSTMCell(self.lstm_dim))

                decoder_layer, _, _ = rnn.stack_bidirectional_dynamic_rnn(
                    forward_cells, backward_cells, encoder_layer, dtype=tf.float32)

            rnn_reshaped = tf.reshape(decoder_layer, [batch_size * width, -1])

            logits = slim.fully_connected(rnn_reshaped, self.num_classes, activation_fn=None)
            logits = tf.reshape(logits, [batch_size, width, self.num_classes])
            rnn_out = tf.transpose(logits, (1, 0, 2))

        return rnn_out
