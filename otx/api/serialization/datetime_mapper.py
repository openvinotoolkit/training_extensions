"""This module contains the mapper for datetime."""

#
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


import datetime
from typing import Union

from otx.api.utils.time_utils import now


class DatetimeMapper:
    """This class maps a `datetime.datetime` entity to a string, and vice versa."""

    @staticmethod
    def forward(instance: datetime.datetime) -> str:
        """Serializes datetime to str."""

        return instance.strftime("%Y-%m-%dT%H:%M:%S.%f")

    @staticmethod
    def backward(instance: Union[None, str]) -> datetime.datetime:
        """Deserializes datetime from str or create new one if it is None."""

        if isinstance(instance, str):
            modification_date = datetime.datetime.strptime(instance, "%Y-%m-%dT%H:%M:%S.%f")
            return modification_date.replace(tzinfo=datetime.timezone.utc)

        return now()
