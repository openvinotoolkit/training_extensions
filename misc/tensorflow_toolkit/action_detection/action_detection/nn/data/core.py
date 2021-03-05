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

from os.path import exists, dirname, realpath, isabs, join
from collections import namedtuple

import tensorflow as tf

ImageSize = namedtuple('ImageSize', 'h, w, c')


def parse_text_records(input_data_file_path, types):
    """Converts input text records into tuple of specified types.

    :param input_data_file_path: Input text file with records
    :param types: Output record types
    :return:
    """

    def _encode(input_value, dtype=None):
        """Converts input value into specified data type.

        :param input_value: Input value
        :param dtype: Output data type
        :return: Encoded value
        """

        if dtype is None:
            return input_value
        elif isinstance(dtype, str):
            if dtype == 'path':
                return input_value if exists(input_value) else None
            else:
                raise Exception('Unknown dtype for record: {}'.format(dtype))
        elif callable(dtype):
            return dtype(input_value)
        else:
            raise Exception('Cannot convert to dtype: {}'.format(dtype))

    assert len(types) > 0
    data_dir = dirname(realpath(input_data_file_path))

    out_data = [[] for _ in xrange(len(types))]
    with open(input_data_file_path, 'r') as input_stream:
        for line in input_stream:
            if line.endswith('\n'):
                line = line[:-len('\n')]

            if len(line) == 0:
                continue

            line_data = line.split(' ')
            assert len(line_data) == len(types)

            encoded_data = []
            valid_record = True
            for i in xrange(len(types)):
                if not isabs(line_data[i]):
                    line_data[i] = join(data_dir, line_data[i])
                encoded_value = _encode(line_data[i], types[i])
                if encoded_value is None:
                    valid_record = False
                    break

                encoded_data.append(encoded_value)

            if valid_record:
                for i in xrange(len(types)):
                    out_data[i].append(encoded_data[i])

    return out_data


def decode_jpeg(image_file_path, num_channels):
    """Loads image tensor from file in .jpeg format

    :param image_file_path: Path to image
    :param num_channels: Input number of channels
    :return:
    """

    image_bytes = tf.read_file(image_file_path)
    image = tf.image.decode_jpeg(image_bytes, channels=num_channels)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def encode_image(src_image, out_height, out_width, to_bgr=True, scale=255.0):
    """Converts raw image tensor into internal network format.

    :param src_image: Source image tensor
    :param out_height: Out image height
    :param out_width: Out image width
    :param to_bgr: Whether to convert into BGR channel format
    :param scale: Scale parameter
    :return:
    """

    blob = tf.image.resize_images(src_image, [out_height, out_width])

    if to_bgr:
        blob = tf.reverse(blob, [-1])  # Convert to BGR format

    if scale != 1.0:
        blob *= scale

    return blob
