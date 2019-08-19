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

import tensorflow as tf

EMBEDDINGS_DIM = 256
WEIGHT_DECAY = 0.0001


def l2_normalized_embeddings(inputs):
    output = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    output = tf.reshape(output, [-1, 1, 1, output.shape[-1]])
    output = tf.keras.layers.Conv2D(
        filters=EMBEDDINGS_DIM, kernel_size=1, padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(l=WEIGHT_DECAY))(output)

    output = tf.reshape(output, [-1, EMBEDDINGS_DIM])
    output = tf.nn.l2_normalize(output * 1000, axis=1, epsilon=1e-13)
    return output


def keras_applications_mobilenetv2(inputs, alpha=1.0):
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

    base_model = MobileNetV2(alpha=alpha, include_top=False,
                             weights='imagenet', input_tensor=inputs)

    output = l2_normalized_embeddings(base_model.output)
    return tf.keras.Model(inputs, output)


def keras_applications_resnet50(inputs):
    from tensorflow.keras.applications.resnet50 import ResNet50

    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)
    output = l2_normalized_embeddings(base_model.output)

    return tf.keras.Model(inputs, output)
