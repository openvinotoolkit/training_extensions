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


class CentralCrop(object):
    """Functor to extract central squared crop of image.
    """

    def __init__(self, side_size, trg_image_size):
        """Constructor.

        :param side_size: Size of crop side
        :param trg_image_size: Target image size
        """

        self._side_size = side_size
        self._trg_image_size = trg_image_size

    def __call__(self, src_image):
        """Carry out image cropping.

        :param src_image: Input image
        :return: Cropped image
        """

        src_image_size = tf.shape(src_image)
        src_height = src_image_size[0]
        src_width = src_image_size[1]
        src_aspect_ratio = tf.cast(src_height, tf.float32) / tf.cast(src_width, tf.float32)

        resized_image =\
            tf.cond(tf.less(src_height, src_width),
                    lambda: tf.image.resize_images(
                        src_image, [self._side_size, tf.cast(float(self._side_size) / src_aspect_ratio, tf.int32)]),
                    lambda: tf.image.resize_images(
                        src_image, [tf.cast(float(self._side_size) * src_aspect_ratio, tf.int32), self._side_size]))

        resized_image_shape = tf.shape(resized_image)
        offset_height =\
            tf.cast(0.5 * tf.cast(tf.abs(resized_image_shape[0] - self._trg_image_size.h), tf.float32), tf.int32)
        offset_width =\
            tf.cast(0.5 * tf.cast(tf.abs(resized_image_shape[1] - self._trg_image_size.w), tf.float32), tf.int32)

        trg_image = tf.image.crop_to_bounding_box(resized_image,
                                                  offset_height, offset_width,
                                                  self._trg_image_size.h, self._trg_image_size.w)

        return trg_image


class ImageNetProcessFn(object):
    """ImageNet-specific image pre-processing functor.
    """

    def __init__(self, central_fraction, trg_image_size):
        """Constructor.

        :param central_fraction: Fraction of central crop
        :param trg_image_size: Target image size
        """

        self._central_fraction = central_fraction
        self._trg_image_size = trg_image_size

    def __call__(self, src_image):
        """Carry out image pre-processing.

        :param src_image: Input image
        :return: Processed image
        """

        trg_image = tf.image.central_crop(src_image, central_fraction=self._central_fraction)

        trg_image = tf.expand_dims(trg_image, 0)
        trg_image = tf.image.resize_bilinear(trg_image, [self._trg_image_size.h, self._trg_image_size.w],
                                             align_corners=False)
        trg_image = tf.squeeze(trg_image, [0])

        return trg_image
