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

from ote.interfaces.dataset import IDataset


class IClassificationDataset(IDataset):
    """
    This is a common interface for objects representing an image classification dataset.
    """
    def __getitem__(self, indx) -> dict:
        """
        This method is supposed to return a dataset item by a given index.
        The resulting dict has the following structure:
        {'img': <Numpy array with HWC image>, 'label': integer object category index}

        :param indx: Integer index of the reqired dataset item

        :return: dict with representing classification dataset item
        """

    @abc.abstractmethod
    def get_annotation(self) -> List[dict]:
        """
        Returns a list containing all annotation of classification dataset.
        List item is a dict {'label': <integer object category index>, ... }
        containing object label and other implementation-specific optionall information.
        Element of the list on position i corresponds to the sample returned by __getitem__(i).
        """

    @abc.abstractmethod
    def get_classes(self) -> List[str]:
        """
        Returns a list of object categories in the dataset
        :return: List of classes
        """
