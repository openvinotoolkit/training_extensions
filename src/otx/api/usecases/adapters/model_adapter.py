"""This module define a module to adapt model weights from a data source."""
# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import abc
from typing import Union


class IDataSource:
    """Class that holds a combination of both a repo and a URL which can be used to fetch data."""

    @property
    @abc.abstractmethod
    def data(self):
        """Returns the data of the source."""
        raise NotImplementedError


class ModelAdapter(metaclass=abc.ABCMeta):
    """The ModelAdapter is an adapter is intended to lazily fetch its binary data from a given data source."""

    def __init__(self, data_source: Union[IDataSource, bytes]):
        self.__data_source = data_source

    @property
    def data_source(self):
        """Returns the data source of the adapter."""
        return self.__data_source

    @data_source.setter
    def data_source(self, value: Union[IDataSource, bytes]):
        self.__data_source = value

    @property
    def data(self):
        """Returns the data of the Model."""
        if isinstance(self.__data_source, IDataSource):
            return self.__data_source.data
        if isinstance(self.__data_source, bytes):
            return self.__data_source
        raise ValueError("This model adapter is not properly initialized with a source of data")

    @property
    def from_file_storage(self) -> bool:
        """Returns if the ModelAdapters data comes from the file storage or not.

        This is used in the model repo to know if the data of the model should be saved or not.
        """
        if isinstance(self.data_source, bytes):
            return False
        return True


class ExportableCodeAdapter(ModelAdapter):
    """Adapter intended to lazily fetch raw exportable code data from a given data source."""
