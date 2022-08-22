"""This module tests classes related to image"""

# Copyright (C) 2020-2021 Intel Corporation
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
import tempfile

import cv2
import numpy as np
import pytest

from otx.api.entities.annotation import Annotation
from otx.api.entities.image import Image
from otx.api.entities.shapes.ellipse import Ellipse
from otx.api.entities.shapes.rectangle import Rectangle
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestImage:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_image(self):
        """
        <b>Description:</b>
        To test Image class

        <b>Input data:</b>
        Instances of Image sourced with data or file path to forged image or both
        Rectangle and Ellipse classes instances
        Instances of Annotation made against rectangle and Ellipsis

        <b>Expected results:</b>
        1. Value error is raised
        2. Value error is raised
        3. Created successfully
        4. Created successfully
        5. Calls of the method against both instances return expected strings
        6. Numpy property contains expected data
        7. Numpy property is changed and contains expected data
        8. Called method returns expected value
        9. Value error raises
        10. Value error raises
        11. Called method returns expected value
        12. Height property contains expected data
        13. Width property contains expected data

        <b>Steps</b>
        1. Create Image without parameters
        2. Create Image with both parameters
        3. Create Image with data parameter
        4. Create Image with path to forged image
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
        test_depth = 4
        data0 = np.random.random_sample((test_height, test_width, test_depth))
        data1 = np.random.random_sample((test_height1, test_width1, test_depth))
        d1_data = np.random.random_sample((test_height1,))
        image = np.random.random_sample((test_height, test_width, test_depth))
        image_path = os.path.join(tempfile.gettempdir(), "test_image.png")
        cv2.imwrite(image_path, image)

        with pytest.raises(ValueError):
            Image()

        with pytest.raises(ValueError):
            Image(data=data0, file_path=image_path)

        data_instance = Image(data=data0)
        assert isinstance(data_instance, Image)

        fp_instance = Image(file_path=image_path)
        assert isinstance(fp_instance, Image)

        assert str(data_instance) == f"Image(with data, width={test_width}, height={test_height})"
        assert str(fp_instance) == f"Image({image_path}, width={test_width}, height={test_height})"

        assert np.array_equal(data_instance.numpy, data0)
        height, width, depth = fp_instance.numpy.shape
        assert height == test_height
        assert width == test_width
        assert depth != test_depth

        data_instance.numpy = data1
        fp_instance.numpy = data1

        assert np.array_equal(data_instance.numpy, data1)
        assert np.array_equal(fp_instance.numpy, data1)

        assert np.array_equal(data_instance.roi_numpy(), data1)
        assert np.array_equal(fp_instance.roi_numpy(), data1)

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

        for arr in data_instance.roi_numpy(rect_annotation):
            for it in arr:
                assert it in data1

        assert data_instance.height == fp_instance.height == test_height1
        assert data_instance.width == fp_instance.width == test_width1

        if os.path.exists(image_path):
            os.remove(image_path)
