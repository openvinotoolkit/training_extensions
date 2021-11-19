"""This module tests classes related to image"""

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


import pytest
import numpy as np
import cv2
import tempfile
import os


from ote_sdk.entities.image import Image
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements
from ote_sdk.entities.annotation import Annotation
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.entities.shapes.ellipse import Ellipse


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestImage:
    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_image(self):
        """
            <b>Description:</b>
            To test Image class

            <b>Input data:</b>
            Instances of Image class sourced with data or file path to forged image or both
            Rectangle and Ellipse classes instances
            Instances of Annotation class made against rectangle and

            <b>Expected results:</b>
            1. Value error raises
            2. Value error raises
            3. Instantiated successfully
            4. Instantiated successfully
            5. Calls of the method against both instances return expected strings
            6. Numpy property contains expected data
            7. Numpy property changed and contains expected data
            8. Called method returns expected value
            9. Value error raises
            10. Value error raises
            11. Called method returns expected value
            12. Height property contains expected data
            13. Width property contains expected data

            <b>Steps</b>
            1. Instantiate Image class without parameters
            2. Instantiate Image class with both parameters
            3. Instantiate Image class with data parameter
            4. Instantiate Image class with path to forged image
            5. Check __str__ method against data instance and fp instance
            6. Check numpy property against data instance and fp instance
            7. Change numpy property and check it against data instance and fp instance
            8. Check roi_numpy method against data instance and fp instance without Annotation
            9. Check roi_numpy method against data instance and fp instance with Annotation of Ellipsis shape
            10. Check roi_numpy method against 1 dimensional data instance with Annotation of Rectangle shape
            11. Check roi_numpy property against data instance and fp instance with Annotation of Rectangle shape
            12. Check height property against data instance and fp instance
            13. Check width property against data instance and fp instance

        """
        test_height = test_width = 128
        test_height1 = test_width1 = 64
        data0 = np.ndarray(shape=(test_height, test_width, 4), dtype=float, order='C')
        data1 = np.ndarray(shape=(test_height1, test_width1, 4), dtype=float, order='C')
        d1_data = np.ndarray(shape=(test_height,), dtype=float, order='C')
        image = np.zeros((test_height, test_width, 4), dtype=np.uint8)
        image_path = os.path.join(tempfile.gettempdir(), 'test_image.png')
        cv2.imwrite(image_path, image)

        with pytest.raises(ValueError):
            Image()

        with pytest.raises(ValueError):
            Image(data=data0, file_path=image_path)

        data_instance = Image(data=data0)
        assert isinstance(data_instance, Image)

        fp_instance = Image(file_path=image_path)
        assert isinstance(fp_instance, Image)

        assert str(data_instance) == "Image(with data, width=128, height=128)"
        assert str(fp_instance) == f"Image({image_path}, width=128, height=128)"

        assert data_instance.numpy.all() == data0.all()
        assert fp_instance.numpy.all() == image.all()

        data_instance.numpy = fp_instance.numpy = data1

        assert data_instance.numpy.all() == data1.all()
        assert fp_instance.numpy.all() == data1.all()

        assert data_instance.roi_numpy().all() == data1.all()
        assert fp_instance.roi_numpy().all() == data1.all()

        rec_shape = Rectangle(x1=0.0, x2=0.5, y1=0.0, y2=0.5)
        ellipsis_shape = Ellipse(x1=0.0, x2=0.5, y1=0.0, y2=0.5)
        label = []
        rect_annotation = Annotation(rec_shape, label)
        ellipsis_annotation = Annotation(ellipsis_shape, label)

        with pytest.raises(ValueError):
            data_instance.roi_numpy(ellipsis_annotation)

        with pytest.raises(ValueError):
            fp_instance.roi_numpy(ellipsis_annotation)

        d1_data_instance = Image(data=d1_data)

        with pytest.raises(ValueError):
            d1_data_instance.roi_numpy(rect_annotation)

        assert data_instance.roi_numpy(rect_annotation).any() == data1.any()
        assert fp_instance.roi_numpy(rect_annotation).any() == data1.any()

        assert data_instance.height == fp_instance.height == test_height1
        assert data_instance.width == fp_instance.width == test_width1

        if os.path.exists(image_path):
            os.remove(image_path)
