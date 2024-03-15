# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


import pickle  # nosec B403

import pytest
from bson import ObjectId

from otx.api.entities.id import ID
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestPickle:
    @pytest.mark.priority_medium
    @pytest.mark.unit
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
        unpickled_id = pickle.loads(pickled_id)  # nosec B301
        assert id(original_id) != id(pickled_id), "Expected two different memory instanced"
        assert original_id == unpickled_id, "Expected content of entities to be equal"
