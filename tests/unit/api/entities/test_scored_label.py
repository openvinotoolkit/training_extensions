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
from otx.api.entities.scored_label import LabelSource, ScoredLabel
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestScoredLabel:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_scored_label(self):
        """
        <b>Description:</b>
        Check the ScoredLabel can correctly return the value

        <b>Input data:</b>
        LabelEntity

        <b>Expected results:</b>
        Test passes if the results match
        """
        car = LabelEntity(id=ID(123456789), name="car", domain=Domain.DETECTION, is_empty=False)
        person = LabelEntity(id=ID(987654321), name="person", domain=Domain.DETECTION, is_empty=False)
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

        label_source = LabelSource()
        assert car_label.label_source == label_source
        user_name = "User Name"
        car_label.label_source.user_id = user_name
        label_source_with_user = LabelSource(user_id=user_name)
        assert car_label.label_source == label_source_with_user

        car.color = Color(red=16, green=15, blue=56, alpha=255)
        assert repr(car_label) == (
            "ScoredLabel(123456789, name=car, probability=0.4, domain=DETECTION, color="
            "Color(red=16, green=15, blue=56, alpha=255), hotkey=, "
            "label_source=LabelSource(user_id='User Name', model_id=ID(), "
            "model_storage_id=ID()))"
        )
