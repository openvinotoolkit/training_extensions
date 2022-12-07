# Copyright (C) 2021 Intel Corporation
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

import pytest

from otx.api.entities.color import Color
from otx.api.entities.id import ID
from otx.api.entities.label import Domain, LabelEntity
from otx.api.utils.time_utils import now
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestDomain:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_domain(self):
        """
        <b>Description:</b>
        Check the Domain can correctly return the value

        <b>Expected results:</b>
        Test passes if the results match
        """
        domain = Domain
        assert len(domain) == 11


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestLabelEntity:

    creation_date = now()

    label_car_params = {
        "name": "car",
        "domain": Domain.DETECTION,
        "color": Color(255, 0, 0),
        "hotkey": "ctrl+1",
        "creation_date": creation_date,
        "is_empty": False,
        "id": ID(123456789),
    }

    label_person_params = {
        "name": "person",
        "domain": Domain.DETECTION,
        "color": Color(255, 17, 17),
        "hotkey": "ctrl+2",
        "creation_date": creation_date,
        "is_empty": False,
        "id": ID(987654321),
    }
    car = LabelEntity(**label_car_params)  # type: ignore
    empty = LabelEntity(name="empty", domain=Domain.SEGMENTATION, is_empty=True)
    person = LabelEntity(**label_person_params)  # type: ignore

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_entity(self):
        """
        <b>Description:</b>
        Check the LabelEntity can correctly return the value

        <b>Input data:</b>
        Dummy data

        <b>Expected results:</b>
        Test passes if incoming data is processed correctly

        <b>Steps</b>
        1. Use already created dummy data
        2. Check the processing of default values
        3. Check the processing of changed values
        """

        assert self.car == LabelEntity(**self.label_car_params)
        assert self.car != Domain
        assert self.car != self.person

        for attr in [
            "name",
            "domain",
            "color",
            "hotkey",
            "creation_date",
            "is_empty",
            "id",
        ]:
            assert getattr(self.car, attr) == self.label_car_params[attr]

        label_car_new_name = "electric car"
        label_car_new_domain = Domain.CLASSIFICATION
        label_car_new_color = Color(0, 255, 0)
        label_car_new_hotkey = "ctrl+2"
        label_car_new_id = ID(987654321)

        setattr(self.car, "name", label_car_new_name)
        setattr(self.car, "domain", label_car_new_domain)
        setattr(self.car, "color", label_car_new_color)
        setattr(self.car, "hotkey", label_car_new_hotkey)
        setattr(self.car, "id", label_car_new_id)

        assert self.car.name == label_car_new_name
        assert self.car.domain == label_car_new_domain
        assert self.car.color == label_car_new_color
        assert self.car.hotkey == label_car_new_hotkey
        assert self.car.id_ == label_car_new_id

        test_label_entity_repr = [
            f"{self.car.id_}",
            f"name={self.car.name}",
            f"hotkey={self.car.hotkey}",
            f"domain={self.car.domain}",
            f"color={self.car.color}",
        ]

        for i in test_label_entity_repr:
            assert i in self.car.__repr__()

        assert hash(self.car) == hash(str(self.car))
        assert self.car.__lt__(Domain) is False
        assert self.car.__gt__(Domain) is False

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_empty_label_entity(self):
        """
        <b>Description:</b>
        Check the LabelEntity can correctly return the value for empty label

        <b>Input data:</b>
        Dummy data

        <b>Expected results:</b>
        Test passes if incoming data is processed correctly

        <b>Steps</b>
        1. Use already created dummy data
        2. Check the processing of default values
        """

        assert self.empty.hotkey == ""
        assert self.empty.id_ == ID()
        assert type(self.empty.color) == Color

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_comparison(self):
        """
        <b>Description:</b>
        Check the LabelEntity __lt__, __gt__ methods with changed id

        <b>Input data:</b>
        Dummy data

        <b>Expected results:</b>
        Test passes if incoming data is processed correctly

        <b>Steps</b>
        1. Use already created dummy data
        2. Check the processing of changed id
        """

        self.empty.id_ = ID(999999999)
        assert self.empty > self.car
        assert self.car < self.empty
