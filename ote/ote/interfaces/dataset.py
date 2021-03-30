"""
 Copyright (c) 2021 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import abc
from typing import List


class IDataset(metaclass=abc.ABCMeta):
    """
    This is a common interface for objects representing a dataset.
    """

    def __getitem__(self, indx) -> dict:
        """
        This method is supposed to return a dataset item by a given index.
        Final layout of an item is defined by task-specific implementation

        :param indx: Integer index of the reqired dataset item

        :return: dict containing a training sample and annotation in a task-specific format
        """
        pass

    def __len__(self) -> int:
        """
        :return: Length of the dataset
        """
        pass

    @abc.abstractmethod
    def get_annotation(self) -> list:
        """
        Returns a list containing all annotation items
        :return: List of annotations
        """
        pass
