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

""" This module contains the mapper for datetime """

import datetime
from typing import Union

from ote_sdk.utils.time_utils import now


class DatetimeMapper:
    """
    This class maps a `datetime.datetime` entity to a string, and vice versa
    """

    @staticmethod
    def forward(instance: datetime.datetime) -> str:
        """Serializes datetime to str."""

        return instance.strftime("%Y-%m-%dT%H:%M:%S.%f")

    @staticmethod
    def backward(instance: Union[None, str]) -> datetime.datetime:
        """Deserializes datetime from str or create new one if it is None"""

        if isinstance(instance, str):
            modification_date = datetime.datetime.strptime(
                instance, "%Y-%m-%dT%H:%M:%S.%f"
            )
            return modification_date.replace(tzinfo=datetime.timezone.utc)

        return now()
