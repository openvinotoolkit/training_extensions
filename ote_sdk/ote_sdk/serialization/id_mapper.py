#
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
#

""" This module contains the mapper for ID entities """

from ote_sdk.entities.id import ID


class IDMapper:
    """
    This class maps an `ID` entity to a string, and vice versa
    """

    @staticmethod
    def forward(instance: ID) -> str:
        """Serializes ID to str."""

        return str(instance)

    @staticmethod
    def backward(instance: str) -> ID:
        """Deserializes ID from str."""

        return ID(str(instance))
