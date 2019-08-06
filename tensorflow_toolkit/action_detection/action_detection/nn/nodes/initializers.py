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

import numpy as np
import tensorflow as tf


def orthogonal_initializer(mode=None, dtype=tf.float32, out_scale=1.0):
    """Generates orthogonal initializer for convolution and dense layers.

    :param mode: Type of operation: dense or conv
    :param dtype: Type of data
    :param out_scale: Scalar value tu multiply on
    :return: Initializer function
    """

    def _generate_ort_vectors(num_in, num_out):
        """Generates random vectors of the specified sizes.

        :param num_in: Input size
        :param num_out: Output size
        :return: Orthogonal vectors
        """

        flat_shape = (num_in, num_out)

        init_stddev = np.sqrt(1.0 / float(num_in)) / .87962566103423978
        flatten_array = np.random.normal(0., init_stddev, flat_shape)

        if num_out > num_in:
            out_weights = flatten_array
        else:
            ort_matrix, _, _ = np.linalg.svd(flatten_array, full_matrices=False)
            assert ort_matrix.shape == flat_shape

            diff = np.matmul(ort_matrix.T, ort_matrix) - np.eye(ort_matrix.shape[1], ort_matrix.shape[1])
            np.testing.assert_array_almost_equal(diff, np.zeros_like(diff))

            out_weights = ort_matrix
            out_weights *= out_scale * init_stddev / np.std(out_weights)

        return out_weights

    def _initializer(shape, dtype=dtype, partition_info=None):
        """Returns orthogonal weights of the specified shape.

        :param shape: Target shape of weights
        :param dtype: Type of data
        :param partition_info: Unused interface parameter
        :return: Initialized weights
        """

        target_mode = mode
        if target_mode is None:
            if len(shape) == 2:
                target_mode = 'dense'
            elif len(shape) == 4:
                target_mode = 'conv'
            else:
                raise Exception('Unsupported shape: {}'.format(shape))
        else:
            assert target_mode in ['dense', 'conv']

        if target_mode == 'dense':
            num_in = shape[0]
            num_out = shape[1]
        else:
            if shape[3] == 1:
                num_in = shape[0] * shape[1]
                num_out = shape[2]
            else:
                num_in = shape[0] * shape[1] * shape[2]
                num_out = shape[3]

        values = _generate_ort_vectors(num_in, num_out)
        weights = tf.constant(values.reshape(shape), dtype=dtype)

        return weights

    return _initializer
