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
