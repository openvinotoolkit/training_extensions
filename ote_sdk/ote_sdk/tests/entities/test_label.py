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
import datetime

from ote_sdk.entities.label import Domain, LabelEntity
from ote_sdk.entities.id import ID
from ote_sdk.entities.color import Color

from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements

@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestDomain:

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_domain(self):
        """
        <b>Description:</b>
        Check that Domain can correctly return the value

        <b>Expected results:</b>
        Test passes if the results matches
        """
        domain = Domain
        assert len(domain) == 6

@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestLabelEntity:

    label_car_params = {
        "name": "car",
        "domain": Domain.DETECTION,
        "color": "#ff0000",
        "hotkey": "ctrl+1",
        "creation_date": datetime.datetime.today(),
        "is_empty": False,
        "id": 123456789,
    }

    other_label_car_params = {
        "name": "person",
        "domain": Domain.DETECTION,
        "color": "#ff1111",
        "hotkey": "ctrl+2",
        "creation_date": datetime.datetime.today(),
        "is_empty": False,
        "id": 987654321,
    }

    car = LabelEntity(**label_car_params) #type: ignore
    empty = LabelEntity(name="empty", domain=Domain.SEGMENTATION, is_empty=True)
    person = LabelEntity(**other_label_car_params) #type: ignore

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_entity(self):
        """
        <b>Description:</b>
        Check that LabelEntity can correctly return the value

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

        for attr in ["name", "domain", "color", "hotkey", "creation_date", "is_empty", "id"]:
            assert getattr(self.car, attr) == self.label_car_params[attr]

        label_car_new_name = "electric car"
        label_car_new_domain = Domain.CLASSIFICATION
        label_car_new_color = "#00ff00"
        label_car_new_hotkey = "ctrl+2"
        label_car_new_id = 987654321

        setattr(self.car, "name", label_car_new_name)
        setattr(self.car, "domain", label_car_new_domain)
        setattr(self.car, "color", label_car_new_color)
        setattr(self.car, "hotkey", label_car_new_hotkey)
        setattr(self.car, "id", label_car_new_id)

        assert self.car.name == label_car_new_name
        assert self.car.domain == label_car_new_domain
        assert self.car.color == label_car_new_color
        assert self.car.hotkey == label_car_new_hotkey
        assert self.car.id == label_car_new_id

        test_label_entity_repr = [
            f"{self.car.id}",
            f"name={self.car.name}",
            f"hotkey={self.car.hotkey}",
            f"domain={self.car.domain}",
            f"color={self.car.color}"

        ]

        for i in test_label_entity_repr:
            assert i in self.car.__repr__()

        assert hash(self.car) == hash(str(self.car))
        assert self.car.__lt__(Domain) == False
        assert self.car.__gt__(Domain) == False

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_empty_label_entity(self):
        """
        <b>Description:</b>
        Check that LabelEntity can correctly return the value for empty label

        <b>Input data:</b>
        Dummy data

        <b>Expected results:</b>
        Test passes if incoming data is processed correctly

        <b>Steps</b>
        1. Use already created dummy data
        2. Check the processing of default values
        """

        assert self.empty.hotkey == "ctrl+0"
        assert self.empty.id == ID()
        assert type(self.empty.color) == Color

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_label_comparison(self):
        """
        <b>Description:</b>
        Check that LabelEntity comparison

        <b>Input data:</b>
        Dummy data

        <b>Expected results:</b>
        Test passes if incoming data is processed correctly

        <b>Steps</b>
        1. Use already created dummy data
        2. Check the processing of shanged id
        """

        self.empty.id = 999999999
        assert self.empty > self.car
        assert self.car < self.empty
