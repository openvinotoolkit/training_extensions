"""
This module contains the interface class for tasks to be notified when the task does not need to be loaded anymore.
"""


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

import abc


class IUnload(metaclass=abc.ABCMeta):
    """
    This interface can be implemented by a task, if the task wants to be notified when
    the task is not needed by the pipeline anymore.
    This allows to clear GPU and system memory resources for example.
    """

    @abc.abstractmethod
    def unload(self):
        """
        Unload any resources which have been used by the task.

        It is acceptable to restart the server as a last resort strategy if unloading the resources is too difficult.
        """
        raise NotImplementedError
