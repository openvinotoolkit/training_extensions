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

from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.label import Domain, LabelEntity
from ote_sdk.entities.color import Color

from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements

@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestScoredLabel:

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_scored_label(self):
        """
        <b>Description:</b>
        Check that ScoredLabel can correctly return the value

        <b>Input data:</b>
        LabelEntity

        <b>Expected results:</b>
        Test passes if the results matches
        """
        car = LabelEntity(id=123456789, name="car", domain=Domain.DETECTION, is_empty=True)
        person = LabelEntity(id=987654321, name="person", domain=Domain.DETECTION, is_empty=True)
        car_label = ScoredLabel(car)
        person_label = ScoredLabel(person)

        for attr in ["id", "name", "color", "hotkey", "creation_date", "is_empty"]:
            assert getattr(car_label, attr) == getattr(car, attr)

        assert car_label.get_label() == car
        assert car_label == ScoredLabel(car)
        assert car_label != car
        assert car_label != person_label
        assert hash(car_label) == hash(str(car_label))

        probability = 0.0
        assert car_label.probability == probability
        delta_probability = 0.4
        probability += delta_probability
        car_label.probability += delta_probability
        assert car_label.probability == probability

        car.color = Color(red=16, green=15, blue=56, alpha=255)
        assert repr(car_label) == "ScoredLabel(123456789, name=car, probability=0.4, domain=DETECTION, color=Color(red=16, green=15, blue=56, alpha=255), hotkey=ctrl+0)"
