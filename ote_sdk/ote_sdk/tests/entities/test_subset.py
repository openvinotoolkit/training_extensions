"""This module tests classes related to Subset"""

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

from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements
from ote_sdk.entities.subset import Subset


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestSubset:
    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_subset_members(self):
        """
        <b>Description:</b>
        To test Subset enumeration members

        <b>Input data:</b>
        Initialized instance of Subset enum

        <b>Expected results:</b>
        Enum members return correct values:

        NONE = 0
        TRAINING = 1
        VALIDATION = 2
        TESTING = 3
        UNLABELED = 4
        PSEUDOLABELED = 5

        <b>Steps</b>
        1. Create enum instance
        2. Check members
        """
        test_instance = Subset

        for i in range(0, 6):
            assert test_instance(i) in list(Subset)

        with pytest.raises(AttributeError):
            test_instance.WRONG

        with pytest.raises(ValueError):
            test_instance(6)

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_subset_magic_str(self):
        """
        <b>Description:</b>
        To test Subset __str__ method

        <b>Input data:</b>
        Initialized instance of Subset enum

        <b>Expected results:</b>
        __str__ return correct string for every enum member
        In case incorrect member it raises attribute exception

        <b>Steps</b>
<<<<<<< HEAD
        1. Create enum instance
=======
        1. Initiate enum instance
>>>>>>> acc788325eabbffc45c67600b7218506799a8fb7
        2. Check returning value of __str__ method
        """
        test_instance = Subset
        magic_str_list = [str(i) for i in list(Subset)]

        for i in range(0, 6):
            assert str(test_instance(i)) in magic_str_list

        with pytest.raises(AttributeError):
            str(test_instance.WRONG)

        with pytest.raises(ValueError):
            str(test_instance(6))

        assert len(set(magic_str_list)) == len(magic_str_list)

    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_subset_magic_repr(self):
        """
        <b>Description:</b>
        To test Subset __repr__ method

        <b>Input data:</b>
        Initialized instance of Subset enum

        <b>Expected results:</b>
        __repr__ method return correct string

        <b>Steps</b>
<<<<<<< HEAD
        1. Create enum instance
=======
        1. Initiate enum instance
>>>>>>> acc788325eabbffc45c67600b7218506799a8fb7
        2. Check returning value of magic methods
        """
        test_instance = Subset
        magic_repr_list = [repr(i) for i in list(Subset)]

        for i in range(0, 6):
            assert repr(test_instance(i)) in magic_repr_list

        with pytest.raises(AttributeError):
            repr(test_instance.WRONG)

        with pytest.raises(ValueError):
            repr(test_instance(6))

        assert len(set(magic_repr_list)) == len(magic_repr_list)
