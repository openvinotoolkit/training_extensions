# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from ote_sdk.entities.url import URL
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestURL:
    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_annotation_repo(self):
        """
        <b>Description:</b>
        Check that URL can correctly decode a given URL

        <b>Input data:</b>
        Some URL strings

        <b>Expected results:</b>
        Test passes if the correct scheme and path can be retrieved from the url

        <b>Steps</b>
        1. Create URL instances
        2. Check scheme of URL
        3. Check path of URL
        """
        test_url = URL("binaryrepo:/images/file_%20_whatever.jpg")
        test_url2 = URL(
            "binaryrepo://intel.com/images/file_%20_whatever.jpg?blaat=none"
        )

        assert test_url.scheme == "binaryrepo"
        assert test_url2.scheme == "binaryrepo"

        assert test_url.path == "/images/file_%20_whatever.jpg"
        assert test_url2.path == "/images/file_%20_whatever.jpg"
