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


import pickle

import pytest
from bson import ObjectId

from ote_sdk.entities.id import ID
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestPickle:
    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_pickle_id(self):
        """
        <b>Description:</b>
        Check ID can be correctly pickled and unpickled

        <b>Input data:</b>
        ID

        <b>Expected results:</b>
        Test passes if the ID is correctly unpickled

        <b>Steps</b>
        1. Create ID
        2. Pickle and unpickle ID
        3. Check ID against unpickled ID
        """
        original_id = ID(ObjectId())
        pickled_id = pickle.dumps(original_id)
        unpickled_id = pickle.loads(pickled_id)
        assert id(original_id) != id(
            pickled_id
        ), "Expected two different memory instanced"
        assert original_id == unpickled_id, "Expected content of entities to be equal"
