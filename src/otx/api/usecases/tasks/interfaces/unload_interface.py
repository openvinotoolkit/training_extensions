"""Unload Interface.

This module contains the interface class for tasks to be notified when the task does not need to be loaded anymore.
"""


# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import abc


class IUnload(metaclass=abc.ABCMeta):
    """Interface to provide unload functionality.

    This interface can be implemented by a task, if the task wants to be notified when
    the task is not needed by the pipeline anymore.
    This allows to clear GPU and system memory resources for example.
    """

    @abc.abstractmethod
    def unload(self):
        """Unload task.

        Unload any resources which have been used by the task.

        It is acceptable to restart the server as a last resort strategy if unloading the resources is too difficult.
        """
        raise NotImplementedError
