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

import logging
from typing import Union

from bson import ObjectId
from ote_sdk.entities.id import ID


class IDMapper:
    """
    This class maps an `ID` entity to a string, and vice versa
    """

    def forward(
        self,
        instance: ID
    ) -> Union[ObjectId, str]:
        instance = ID(instance)
        # if len(instance) in (12, 24):
        #     return ObjectId(str(instance))
        # if str(instance).isdigit():
        #     # ID exists of digits only. Pad with zero's to a string of length 24 (e.g.19 -> "..0019")
        #     id_int = int(str(instance))
        #     return ObjectId("{:024d}".format(id_int))

        if len(instance) != 0:
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Warning: Using str instead of ObjectId for {str(instance)}"
            )
        return str(instance)

    def backward(
        self,
        instance: str
    ) -> ID:
        return ID(str(instance))
