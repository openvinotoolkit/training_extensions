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
import abc

from ote_sdk.entities.media import IMediaEntity, IMedia2DEntity
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestIMediaEntity:
    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_i_media_entity(self):
        """
        <b>Description:</b>
        To test IMediaEntity class

        <b>Input data:</b>
        Instance of IMediaEntity class

        <b>Expected results:</b>
        1. Test instance is instance of class IMediaEntity


        <b>Steps</b>
        1. Instantiate an instance of class

        """
        test_inst = IMediaEntity()
        assert isinstance(test_inst, IMediaEntity)


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestIMedia2DEntity:
    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_i_media_2d_entity(self):
        """
        <b>Description:</b>
        To test IMedia2DEntity abstract class

        <b>Input data:</b>
        Instance of IMedia2DEntity abstract class

        <b>Expected results:</b>
        1. TypeError is rose
        2. Expected method numbers and names


        <b>Steps</b>
        1. Instantiate an instance of abstract class
        2. Check abstract methods

        """
        with pytest.raises(TypeError):
            IMedia2DEntity()

        assert type(IMedia2DEntity) is abc.ABCMeta
        abc_methods = IMedia2DEntity.__abstractmethods__
        assert len(abc_methods) == 4
        assert 'width' in abc_methods
        assert 'roi_numpy' in abc_methods
        assert 'numpy' in abc_methods
        assert 'height' in abc_methods
