#
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
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
