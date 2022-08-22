# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from otx.api.entities.url import URL
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestURL:
    @pytest.mark.priority_medium
    @pytest.mark.unit
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
        test_url2 = URL("binaryrepo://intel.com/images/file_%20_whatever.jpg?blaat=none")

        assert test_url.scheme == "binaryrepo"
        assert test_url2.scheme == "binaryrepo"

        assert test_url.path == "/images/file_%20_whatever.jpg"
        assert test_url2.path == "/images/file_%20_whatever.jpg"

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_url_initialization(self):
        """
        <b>Description:</b>
        Check initialized URL object attributes

        <b>Input data:</b>
        Initialized URL object

        <b>Expected results:</b>
        Test passes if attributes of initialized URL object are equal to expected

        <b>Steps</b>
        1. Check attributes of URL object initialized without special characters
        2. Check attributes of URL object initialized with "#" special character
        3. Check attributes of URL object initialized with "?" special character
        4. Check attributes of URL object initialized with netloc
        5. Check attributes of URL object initialized with unsupported character in URL-scheme
        6. Check attributes of URL object initialized without URL-scheme
        7. Check that ValueError exception is raised when initializing URL-class object with incorrect netloc
        """

        def check_url(
            url: URL,
            expected_path: str = "",
            expected_extension: str = "",
            expected_host: str = "",
            expected_scheme: str = "",
            expected_relative_path: str = "",
        ):
            assert url.path == expected_path
            assert url.extension == expected_extension
            assert url.host == expected_host
            assert url.scheme == expected_scheme
            if expected_relative_path == "":
                expected_relative_path = expected_path[1:]
            assert url.relative_path == expected_relative_path

        # Checking attributes of URL initialized without special characters
        check_url(
            url=URL("binaryrepo:/images/file_%20_whatever.jpg"),
            expected_path="/images/file_%20_whatever.jpg",
            expected_extension="jpg",
            expected_scheme="binaryrepo",
        )
        # Checking attributes of URL initialized with "#" special character
        check_url(
            url=URL("binaryrepo:/images/file_%20_whatever#.jpg"),
            expected_path="/images/file_%20_whatever",
            expected_scheme="binaryrepo",
        )
        # Checking attributes of URL initialized with "?" special character
        check_url(
            url=URL("binaryrepo:/i?mages/file_%20_whatever.jpg?some_text.png"),
            expected_path="/i",
            expected_scheme="binaryrepo",
        )
        # Checking attributes of URL initialized with netloc
        check_url(
            url=URL("https://www.some_host.com/images/file_%20_whatever.png"),
            expected_path="/images/file_%20_whatever.png",
            expected_extension="png",
            expected_host="www.some_host.com",
            expected_scheme="https",
        )
        # Checking attributes of URL initialized with unsupported character in URL-scheme
        check_url(
            url=URL("binary_repo:/images/file_%20_whatever.jpg"),
            expected_path="binary_repo:/images/file_%20_whatever.jpg",
            expected_extension="jpg",
            expected_relative_path="binary_repo:/images/file_%20_whatever.jpg",
        )
        # Checking attributes of URL initialized without URL-scheme
        check_url(
            url=URL("images/file_%20_whatever.jpg"),
            expected_path="images/file_%20_whatever.jpg",
            expected_extension="jpg",
            expected_relative_path="images/file_%20_whatever.jpg",
        )
        # Checking that ValueError exception is raised when initializing URL with incorrect netloc
        for incorrect_netloc in [
            "https://[www.some_host.com/images/file_%20_whatever.jpg?some_text.png",
            "https://www.some_host.com]/images/file_%20_whatever.jpg?some_text.png",
        ]:
            with pytest.raises(ValueError):
                URL(incorrect_netloc)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_url_str(self):
        """
        <b>Description:</b>
        Check URL class __str__ method

        <b>Input data:</b>
        Initialized URL object

        <b>Expected results:</b>
        Test passes if string returned by __str__ method is equal to expected

        <b>Steps</b>
        1. Check string returned by __str__ method for URL object initialized with non-empty "scheme" and "host"
        attributes
        2. Check string returned by __str__ method for URL object initialized with non-empty "scheme" and empty "host"
        attributes
        3. Check string returned by __str__ method for URL object initialized with non-empty "host" and empty "scheme"
        attributes
        """
        # Checking value returned by __str__ for URL initialized with non-empty "scheme" and "host"
        assert str(URL("https://www.some_host.com/images/file_%20_whatever.png")) == (
            "https://www.some_host.com/images/file_%20_whatever.png"
        )
        # Checking value returned by __str__ for URL initialized with non-empty "scheme" and empty "host"
        assert str(URL("binaryrepo:/images/file_%20_whatever.jpg")) == "binaryrepo:/images/file_%20_whatever.jpg"
        # Checking value returned by __str__ for URL initialized with non-empty "host" and empty "scheme"
        assert str(URL("//www.some_host.com/images/file_%20_whatever.jpg")) == "/images/file_%20_whatever.jpg"

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_url_repr(self):
        """
        <b>Description:</b>
        Check URL class __repr__ method

        <b>Input data:</b>
        Initialized URL object

        <b>Expected results:</b>
        Test passes if string returned by __repr__ method is equal to expected
        """
        assert repr(URL("https://www.some_host.com/images/file_%20_whatever.png")) == (
            "URL(https://www.some_host.com/images/file_%20_whatever.png)"
        )
        assert repr(URL("binaryrepo:/i?mages/file_%20_whatever.jpg?some_text.png")) == (
            "URL(binaryrepo:/i?mages/file_%20_whatever.jpg?some_text.png)"
        )

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_url_hash(self):
        """
        <b>Description:</b>
        Check URL class __hash__ method

        <b>Input data:</b>
        Initialized URL object

        <b>Expected results:</b>
        Test passes if value returned by __hash__ method is equal to expected
        """
        url = URL("https://www.some_host.com/images/file_%20_whatever.png")
        assert hash(url) == hash(str(url))

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_url_length(self):
        """
        <b>Description:</b>
        Check URL class __len__ method

        <b>Input data:</b>
        Initialized URL object

        <b>Expected results:</b>
        Test passes if value returned by __len__ method is equal to expected
        """
        assert len(URL("https://www.some_host.com/images/file_%20_whatever.png")) == 54
        assert len(URL("binaryrepo:/i?mages/file_%20_whatever.jpg?some_text.png")) == 55

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_url_eq(self):
        """
        <b>Description:</b>
        Check URL class __eq__ method

        <b>Input data:</b>
        Initialized URL object

        <b>Expected results:</b>
        Test passes if value returned by __eq__ method is equal to expected

        1. Check value returned by __eq__ method for URL objects with equal "scheme", "host", and "path" properties
        2. Check value returned by __eq__ method for URL objects with unequal "scheme" properties
        3. Check value returned by __eq__ method for URL objects with unequal "host" properties
        4. Check value returned by __eq__ method for URL objects with unequal "path" properties
        """
        # Checking value returned by __eq__ for URL with equal "scheme", "host", and "path"
        url = URL("https://www.some_host.com/images/file_%20_whatever.png")
        equal_url = URL("https://www.some_host.com/images/file_%20_whatever.png")
        assert url == equal_url
        # Checking value returned by __eq__ for URL with unequal "scheme"
        unequal_url = URL("http://www.some_host.com/images/file_%20_whatever.png")
        assert not url == unequal_url
        # Checking value returned by __eq__ for URL with unequal "host"
        unequal_url = URL("https://www.unequal_host.com/images/file_%20_whatever.png")
        assert not url == unequal_url
        # Checking value returned by __eq__ for URL with unequal "path"
        unequal_url = URL("https://www.some_host.com/images/file_%20_whatever.jpg")
        assert not url == unequal_url
