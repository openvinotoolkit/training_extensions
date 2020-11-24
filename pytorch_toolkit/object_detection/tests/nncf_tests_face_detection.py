# Copyright (C) 2020 Intel Corporation
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

import os

from common.test_case import create_nncf_test_case


class FaceDetection0200NNCFnocompression(
        create_nncf_test_case(
            'face-detection',
            'face-detection-0200',
            os.path.dirname(__file__) + '/../../../data/airport/annotation_faces_train.json',
            os.path.dirname(__file__) + '/../../../data/airport/',
            {
                'compression.int8': False,
                'hyper_parameters.basic.epochs': 2,
                'hyper_parameters.basic.base_learning_rate': 0.00025,
            }
        )
):
    """ Test case for face-detection-0200 model without NNCF compression. """

class FaceDetection0200NNCFint8(
        create_nncf_test_case(
            'face-detection',
            'face-detection-0200',
            os.path.dirname(__file__) + '/../../../data/airport/annotation_faces_train.json',
            os.path.dirname(__file__) + '/../../../data/airport/',
            {'compression.int8': True}
        )
):
    """ Test case for face-detection-0200 model with NNCF int8 compression. """

class FaceDetection0200NNCFint8sparsity(
        create_nncf_test_case(
            'face-detection',
            'face-detection-0200',
            os.path.dirname(__file__) + '/../../../data/airport/annotation_faces_train.json',
            os.path.dirname(__file__) + '/../../../data/airport/',
            {'compression.int8': True, 'compression.sparsity': True}
        )
):
    """ Test case for face-detection-0200 model with NNCF int8 and sparsity compression. """
