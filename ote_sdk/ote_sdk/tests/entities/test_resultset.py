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

from ote_sdk.entities.resultset import ResultSetEntity, ResultsetPurpose
from ote_sdk.entities.metrics import NullPerformance
from ote_sdk.entities.id import ID
from ote_sdk.utils.time_utils import now
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements

@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestResultset:

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_resultset_purpose(self):
        """
        <b>Description:</b>
        Check that ResultsetPurpose can correctly return the value

        <b>Input data:</b>
        Denotes, stages

        <b>Expected results:</b>
        Test passes if the results matches
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
        Check that ResultSetEntity can correctly return the value

        <b>Input data:</b>
        Mock data

        <b>Expected results:</b>
        Test passes if incoming data is processed correctly

        <b>Steps</b>
        1. Create moke data
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
            name = name[0]
            set_attr_name = f"test_{name}"
            if name in [
                "model",
                "ground_truth_dataset",
                "prediction_dataset",
                "purpose"
            ]:
                assert getattr(result_set, name) == value
                setattr(result_set, name, set_attr_name)
                assert getattr(result_set, name) == set_attr_name

        assert result_set.performance == NullPerformance()
        assert result_set.creation_date == now()
        assert result_set.id == ID()

        assert result_set.has_score_metric() == False
        result_set.performance = "test_performance"
        assert result_set.performance != NullPerformance()
        assert result_set.has_score_metric() == True

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
            f"id={result_set.id}"

        ]
        for i in test_result_set_repr:
            assert i in repr(result_set)
