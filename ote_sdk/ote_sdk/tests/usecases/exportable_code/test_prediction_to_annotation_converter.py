# INTEL CONFIDENTIAL
#
# Copyright (C) 2021 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were provided to
# you ("License"). Unless the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit this software or the related documents
# without Intel's prior written permission.
#
# This software and the related documents are provided as is,
# with no express or implied warranties, other than those that are expressly stated
# in the License.

import numpy as np
import pytest

from ote_sdk.entities.label import Domain, LabelEntity
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements

try:
    from ote_sdk.usecases.exportable_code.prediction_to_annotation_converter import (
        DetectionToAnnotationConverter,
    )

    @pytest.mark.components(OteSdkComponent.OTE_SDK)
    class TestPredictionToAnnotationConverter:
        @pytest.mark.priority_medium
        @pytest.mark.component
        @pytest.mark.reqids(Requirements.REQ_1)
        def test_detection_to_annotation_convert(self):
            """
            <b>Description:</b>
            Check that DetectionToAnnotationConverter correctly converts Network output to list of Annotation

            <b>Input data:</b>
            Array of network output with shape [4,6]

            <b>Expected results:</b>
            Test passes if each Converted annotation has the same  values as the network output

            <b>Steps</b>
            1. Create mock network output
            2. Convert network output to Annotation
            3. Check Annotations
            """
            test_boxes = np.array(
                (
                    (0, 0.6, 0.1, 0.1, 0.2, 0.3),
                    (1, 0.2, 0.2, 0.1, 0.3, 0.4),
                    (1, 0.7, 0.3, 0.2, 0.5, 0.6),
                    (0, 0.1, 0.1, 0.1, 0.2, 0.3),
                )
            )

            labels = [
                LabelEntity("Zero", domain=Domain.DETECTION),
                LabelEntity("One", domain=Domain.DETECTION),
            ]

            converter = DetectionToAnnotationConverter(labels)

            annotation_scene = converter.convert_to_annotation(test_boxes)

            for i, annotation in enumerate(annotation_scene.annotations):
                label: ScoredLabel = next(iter(annotation.get_labels()))
                test_label = labels[int(test_boxes[i][0])]
                assert test_label.name == label.name

                assert test_boxes[i][1], label.probability

                assert test_boxes[i][2] == annotation.shape.x1
                assert test_boxes[i][3] == annotation.shape.y1
                assert test_boxes[i][4] == annotation.shape.x2
                assert test_boxes[i][5] == annotation.shape.y2

            annotation_scene = converter.convert_to_annotation(np.ndarray((0, 6)))
            assert 0 == len(annotation_scene.shapes)

        @pytest.mark.priority_medium
        @pytest.mark.component
        @pytest.mark.reqids(Requirements.REQ_1)
        def test_detection_to_annotation_convert_openvino_shape(self):
            """
            <b>Description:</b>
            Check that DetectionToAnnotationConverter correctly converts OpenVINO Network output to annotations

            <b>Input data:</b>
            Array of network output with shape [4,7]

            <b>Expected results:</b>
            Test passes if each Converted annotation has the same  values as the network output

            <b>Steps</b>
            1. Create mock network output
            2. Convert network output to Annotation
            3. Check Annotations
            """
            test_boxes = np.array(
                (
                    (-12, 0, 0.6, 0.1, 0.1, 0.2, 0.3),
                    (12, 1, 0.2, 0.0, 0.1, 0.1, 0.2),
                    (1234, 1, 0.7, 0.2, 0.4, 0.7, 0.5),
                    (1251, 0, 0.1, 0.1, 0.1, 0.2, 0.3),
                )
            )

            labels = [
                LabelEntity("Zero", domain=Domain.DETECTION),
                LabelEntity("One", domain=Domain.DETECTION),
            ]

            converter = DetectionToAnnotationConverter(labels)

            annotation_scene = converter.convert_to_annotation(test_boxes)

            for i, annotation in enumerate(annotation_scene.annotations):
                label: ScoredLabel = next(iter(annotation.get_labels()))
                test_label = labels[int(test_boxes[i][1])]
                assert test_label.name == label.name

                assert test_boxes[i][2] == label.probability

                assert test_boxes[i][3] == annotation.shape.x1
                assert test_boxes[i][4] == annotation.shape.y1
                assert test_boxes[i][5] == annotation.shape.x2
                assert test_boxes[i][6] == annotation.shape.y2

        @pytest.mark.priority_medium
        @pytest.mark.component
        @pytest.mark.reqids(Requirements.REQ_1)
        def test_detection_to_annotation_convert_invalid_input(self):
            """
            <b>Description:</b>
            Check that DetectionToAnnotationConverter raises an error if invalid inputs are provided

            <b>Input data:</b>
            Array of size [1203, 5]
            Array of size [3, 8]

            <b>Expected results:</b>
            Test passes a ValueError is raised for both inputs

            <b>Steps</b>
            1. Create DetectionToAnnotationConverter
            2. Attempt to convert array of [1203,5] to annotations
            3. Attempt to convert array of [3, 8] to annotations
            """
            labels = [
                LabelEntity("Zero", domain=Domain.DETECTION),
                LabelEntity("One", domain=Domain.DETECTION),
            ]
            converter = DetectionToAnnotationConverter(labels)

            with pytest.raises(ValueError):
                converter.convert_to_annotation(np.ndarray((1203, 5)))

            with pytest.raises(ValueError):
                converter.convert_to_annotation(np.ndarray((3, 8)))


except ImportError as e:
    import warnings

    warnings.warn("\nTODO(ikrylov): Add OpenVINO to requirerements.txt!\n" + str(e))
