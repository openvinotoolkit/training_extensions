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

import abc

import pytest

from otx.api.entities.media import IMedia2DEntity, IMediaEntity
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestIMediaEntity:
    @pytest.mark.priority_medium
    @pytest.mark.unit
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
        1. Create IMediaEntity

        """
        test_inst = IMediaEntity()
        assert isinstance(test_inst, IMediaEntity)


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestIMedia2DEntity:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_i_media_2d_entity(self):
        """
        <b>Description:</b>
        To test IMedia2DEntity abstract class

        <b>Input data:</b>
        Instance of IMedia2DEntity abstract class

        <b>Expected results:</b>
        1. TypeError is raised
        2. Expected method numbers and names


        <b>Steps</b>
        1. Create IMedia2DEntity
        2. Check abstract methods

        """
        with pytest.raises(TypeError):
            IMedia2DEntity()

        assert type(IMedia2DEntity) is abc.ABCMeta
        abc_methods = IMedia2DEntity.__abstractmethods__
        assert len(abc_methods) == 4
        assert "width" in abc_methods
        assert "roi_numpy" in abc_methods
        assert "numpy" in abc_methods
        assert "height" in abc_methods
