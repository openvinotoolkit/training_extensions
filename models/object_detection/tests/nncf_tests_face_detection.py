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

from common.object_detection_test_case import create_object_detection_nncf_test_case # pylint: disable=import-error


class FaceDetection0200NNCFint8(
        create_object_detection_nncf_test_case(
            'face-detection',
            'face-detection-0200',
            os.path.dirname(__file__) + '/../../../data/airport/annotation_faces_train.json',
            os.path.dirname(__file__) + '/../../../data/airport/',
            '--nncf-quantization'
        )
):
    """ Test case for face-detection-0200 model with NNCF int8 compression. """

class FaceDetection0200NNCFint8FromTemplate(
        create_object_detection_nncf_test_case(
            'face-detection',
            'face-detection-0200',
            os.path.dirname(__file__) + '/../../../data/airport/annotation_faces_train.json',
            os.path.dirname(__file__) + '/../../../data/airport/',
            '',
            {'optimisations.nncf_quantization.default': 1}
        )
):
    """ Test case for face-detection-0200 model with NNCF int8 compression initialized from template.yaml. """

class FaceDetection0200NNCFint8sparsity(
        create_object_detection_nncf_test_case(
            'face-detection',
            'face-detection-0200',
            os.path.dirname(__file__) + '/../../../data/airport/annotation_faces_train.json',
            os.path.dirname(__file__) + '/../../../data/airport/',
            '--nncf-quantization --nncf-sparsity',
            {},
            {
                'nncf_sparsity.total_epochs': 4
            }
        )
):
    """ Test case for face-detection-0200 model with NNCF int8 and sparsity compression. """

class FaceDetection0202NNCFint8(
        create_object_detection_nncf_test_case(
            'face-detection',
            'face-detection-0202',
            os.path.dirname(__file__) + '/../../../data/airport/annotation_faces_train.json',
            os.path.dirname(__file__) + '/../../../data/airport/',
            '--nncf-quantization'
        )
):
    """ Test case for face-detection-0202 model with NNCF int8 compression. """

class FaceDetection0204NNCFint8(
        create_object_detection_nncf_test_case(
            'face-detection',
            'face-detection-0204',
            os.path.dirname(__file__) + '/../../../data/airport/annotation_faces_train.json',
            os.path.dirname(__file__) + '/../../../data/airport/',
            '--nncf-quantization',
            test_export_threshold=0.21 # TODO(lbeynens): it's very big threshold, try to fix this
        )
):
    """ Test case for face-detection-0204 model with NNCF int8 compression. """

# TODO(lbeynens): fix the test
#class FaceDetection0205NNCFint8(
#        create_object_detection_nncf_test_case(
#            'face-detection',
#            'face-detection-0205',
#            os.path.dirname(__file__) + '/../../../data/airport/annotation_faces_train.json',
#            os.path.dirname(__file__) + '/../../../data/airport/',
#            '--nncf-quantization'
#        )
#):
#    """ Test case for face-detection-0205 model with NNCF int8 compression. """


# # Quantized model is crashing at OpenVINO inference. Disabling the test until the bugfix.
# class FaceDetection0207NNCFint8(
#         create_object_detection_nncf_test_case(
#             'face-detection',
#             'face-detection-0207',
#             os.path.dirname(__file__) + '/../../../data/airport/annotation_faces_train.json',
#             os.path.dirname(__file__) + '/../../../data/airport/',
#             '--nncf-quantization'
#         )
# ):
#     """ Test case for face-detection-0207 model with NNCF int8 compression. """

class PersonDetection0200NNCFint8(
        create_object_detection_nncf_test_case(
            'person-detection',
            'person-detection-0200',
            os.path.dirname(__file__) + '/../../../data/airport/annotation_person_train.json',
            os.path.dirname(__file__) + '/../../../data/airport/train',
            '--nncf-quantization'
        )
):
    """ Test case for person-detection-0200 model with NNCF int8 compression. """

class PersonDetection0201NNCFint8(
        create_object_detection_nncf_test_case(
            'person-detection',
            'person-detection-0201',
            os.path.dirname(__file__) + '/../../../data/airport/annotation_person_train.json',
            os.path.dirname(__file__) + '/../../../data/airport/train',
            '--nncf-quantization'
        )
):
    """ Test case for person-detection-0201 model with NNCF int8 compression. """

class PersonDetection0202NNCFint8(
        create_object_detection_nncf_test_case(
            'person-detection',
            'person-detection-0202',
            os.path.dirname(__file__) + '/../../../data/airport/annotation_person_train.json',
            os.path.dirname(__file__) + '/../../../data/airport/train',
            '--nncf-quantization'
        )
):
    """ Test case for person-detection-0202 model with NNCF int8 compression. """

class PersonVehicleBikeDetection2000NNCFint8(
        create_object_detection_nncf_test_case(
            'person-vehicle-bike-detection',
            'person-vehicle-bike-detection-2000',
            os.path.dirname(__file__) + '/../../../data/airport/annotation_example_train.json',
            os.path.dirname(__file__) + '/../../../data/airport/train',
            '--nncf-quantization'
        )
):
    """ Test case for person-vehicle-bike-detection-2000 model with NNCF int8 compression. """

class PersonVehicleBikeDetection2001NNCFint8(
        create_object_detection_nncf_test_case(
            'person-vehicle-bike-detection',
            'person-vehicle-bike-detection-2001',
            os.path.dirname(__file__) + '/../../../data/airport/annotation_example_train.json',
            os.path.dirname(__file__) + '/../../../data/airport/train',
            '--nncf-quantization'
        )
):
    """ Test case for person-vehicle-bike-detection-2001 model with NNCF int8 compression. """

class PersonVehicleBikeDetection2002NNCFint8(
        create_object_detection_nncf_test_case(
            'person-vehicle-bike-detection',
            'person-vehicle-bike-detection-2002',
            os.path.dirname(__file__) + '/../../../data/airport/annotation_example_train.json',
            os.path.dirname(__file__) + '/../../../data/airport/train',
            '--nncf-quantization'
        )
):
    """ Test case for person-vehicle-bike-detection-2002 model with NNCF int8 compression. """

class VehicleDetection0200NNCFint8(
        create_object_detection_nncf_test_case(
            'vehicle-detection',
            'vehicle-detection-0200',
            os.path.dirname(__file__) + '/../../../data/vehicle_detection/annotation_train.json',
            os.path.dirname(__file__) + '/../../../data/vehicle_detection/train',
            '--nncf-quantization'
        )
):
    """ Test case for vehicle-detection-0200 model with NNCF int8 compression. """

class VehicleDetection0201NNCFint8(
        create_object_detection_nncf_test_case(
            'vehicle-detection',
            'vehicle-detection-0201',
            os.path.dirname(__file__) + '/../../../data/vehicle_detection/annotation_train.json',
            os.path.dirname(__file__) + '/../../../data/vehicle_detection/train',
            '--nncf-quantization'
        )
):
    """ Test case for vehicle-detection-0201 model with NNCF int8 compression. """

class VehicleDetection0202NNCFint8(
        create_object_detection_nncf_test_case(
            'vehicle-detection',
            'vehicle-detection-0202',
            os.path.dirname(__file__) + '/../../../data/vehicle_detection/annotation_train.json',
            os.path.dirname(__file__) + '/../../../data/vehicle_detection/train',
            '--nncf-quantization'
        )
):
    """ Test case for vehicle-detection-0202 model with NNCF int8 compression. """
