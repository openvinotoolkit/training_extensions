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

import datetime

import pytest

from otx.api.entities.id import ID
from otx.api.entities.metrics import NullPerformance
from otx.api.entities.resultset import ResultSetEntity, ResultsetPurpose
from otx.api.utils.time_utils import now
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestResultset:
    creation_date = now()

    @pytest.mark.priority_medium
    @pytest.mark.unit
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
    @pytest.mark.unit
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
        assert type(result_set.creation_date) == datetime.datetime
        assert result_set.id_ == ID()

        assert result_set.has_score_metric() is False
        result_set.performance = "test_performance"
        assert result_set.performance != NullPerformance()
        assert result_set.has_score_metric() is True

        creation_date = self.creation_date
        result_set.creation_date = creation_date
        assert result_set.creation_date == creation_date

        set_attr_id = ID(123456789)
        result_set.id_ = set_attr_id
        assert result_set.id_ == set_attr_id

        test_result_set_repr = [
            f"model={result_set.model}",
            f"ground_truth_dataset={result_set.ground_truth_dataset}",
            f"prediction_dataset={result_set.prediction_dataset}",
            f"purpose={result_set.purpose}",
            f"performance={result_set.performance}",
            f"creation_date={result_set.creation_date}",
            f"id={result_set.id_}",
        ]

        for i in test_result_set_repr:
            assert i in repr(result_set)
