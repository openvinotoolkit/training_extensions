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

""" This module contains the mapper for primitive entities """

import datetime
from typing import Union

from dateutil.tz.tz import tzlocal
from ote_sdk.utils.time_utils import now


class DatetimeMapper:
    """
    This class maps a `datetime.datetime` entity to a string, and vice versa
    """

    def __init__(self) -> None:
        self.tzinfo = tzlocal()

    def forward(
        self,
        instance: datetime.datetime
    ):
        return instance

    def backward(
        self,
        instance: Union[None, str, datetime.datetime]
    ) -> datetime.datetime:

        if isinstance(instance, str):
            # Backward compatibility with the old format.
            # The old format is the ISO format in localtime, however the new format is UTC.
            try:
                modification_date = datetime.datetime.strptime(
                    instance, "%Y-%m-%dT%H:%M:%S.%f"
                )
                modification_date = modification_date.astimezone(datetime.timezone.utc)
            except (ValueError, TypeError):
                modification_date = now()

            return modification_date
        if isinstance(instance, datetime.datetime):
            # Manually insert the timezone.
            return instance.replace(tzinfo=datetime.timezone.utc)
        # Case where instance is None or we received an unexpected type.
        return now()

