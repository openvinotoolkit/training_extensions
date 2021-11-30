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

from datetime import datetime

import pytest

from ote_sdk.entities.id import ID
from ote_sdk.entities.metrics import NullPerformance
from ote_sdk.entities.resultset import ResultSetEntity, ResultsetPurpose
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements
from ote_sdk.utils.time_utils import now


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestResultset:
    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_resultset_purpose(self):
        """
        <b>Description:</b>
        Check the ResultsetPurpose can correctly return the value

        <b>Input data:</b>
        Denotes, stages

        <b>Expected results:</b>
        Test passes if the results match
        """

        denotes = ["EVALUATION", "TEST", "PREEVALUATION"]
        stages = ["Validation", "Test", "Pre-validation"]

        resultset_purpose = ResultsetPurpose
        assert len(resultset_purpose) == 3

        for i in ResultsetPurpose:
            resultset_purpose = ResultsetPurpose(i)
            assert repr(resultset_purpose) == denotes[i.value]
            assert str(resultset_purpose) == stages[i.value]

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_resultset_entity(self):
        """
        <b>Description:</b>
        Check the ResultSetEntity can correctly return the value

        <b>Input data:</b>
        Mock data

        <b>Expected results:</b>
        Test passes if incoming data is processed correctly

        <b>Steps</b>
        1. Create dummy data
        2. Check the processing of default values
        3. Check the processing of changed values
        """

        test_data = {
            "model": None,
            "ground_truth_dataset": None,
            "prediction_dataset": None,
            "purpose": None,
            "performance": None,
            "creation_date": None,
            "id": None,
        }

        result_set = ResultSetEntity(**test_data)

        for name, value in test_data.items():
            set_attr_name = f"test_{name}"
            if name in [
                "model",
                "ground_truth_dataset",
                "prediction_dataset",
                "purpose",
            ]:
                assert getattr(result_set, name) == value
                setattr(result_set, name, set_attr_name)
                assert getattr(result_set, name) == set_attr_name

        assert result_set.performance == NullPerformance()
        assert type(result_set.creation_date) == datetime
        assert result_set.id == ID()

        assert result_set.has_score_metric() is False
        result_set.performance = "test_performance"
        assert result_set.performance != NullPerformance()
        assert result_set.has_score_metric() is True

        result_set.creation_date = now().replace(microsecond=0)
        assert result_set.creation_date == now().replace(microsecond=0)

        set_attr_id = "123456789"
        result_set.id = set_attr_id
        assert result_set.id == set_attr_id

        test_result_set_repr = [
            f"model={result_set.model}",
            f"ground_truth_dataset={result_set.ground_truth_dataset}",
            f"prediction_dataset={result_set.prediction_dataset}",
            f"purpose={result_set.purpose}",
            f"performance={result_set.performance}",
            f"creation_date={result_set.creation_date}",
            f"id={result_set.id}",
        ]

        for i in test_result_set_repr:
            assert i in repr(result_set)
