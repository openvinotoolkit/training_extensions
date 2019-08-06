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


def generate_clustered_prior_boxes(feature_size, image_size, anchor_sizes, step, clip=False, offset=0.5):
    """Generates set of prior boxes according specified anchor sizes.

    :param feature_size: Size of target feature map
    :param image_size: Input size
    :param anchor_sizes: Sizes of anchors in term of image size
    :param step: Step of sliding window
    :param clip: Whether to clip coordinates into [0, 1] range
    :param offset: Parameter to shift coordinates
    :return: Set of anchor boxes
    """

    if isinstance(step, (list, tuple)):
        step_y, step_x = step
    else:
        step_y, step_x = step, step

    height, width = feature_size

    num_priors_per_pixel = len(anchor_sizes)
    num_priors = height * width * num_priors_per_pixel
    top_shape = num_priors, 4

    anchors = []
    for row in xrange(height):
        for col in xrange(width):
            center_x = (float(col) + float(offset)) * float(step_x)
            center_y = (float(row) + float(offset)) * float(step_y)

            for anchor_height, anchor_width in anchor_sizes:
                ymin = (center_y - 0.5 * float(anchor_height)) / float(image_size[0])
                xmin = (center_x - 0.5 * float(anchor_width)) / float(image_size[1])
                ymax = (center_y + 0.5 * float(anchor_height)) / float(image_size[0])
                xmax = (center_x + 0.5 * float(anchor_width)) / float(image_size[1])

                anchors.append([ymin, xmin, ymax, xmax])

    priors_array = np.array(anchors, dtype=np.float32).reshape(top_shape)

    if clip:
        priors_array = np.clip(priors_array, 0., 1.)

    return priors_array, num_priors
