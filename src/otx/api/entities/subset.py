"""This file defines the Subset enum for use in datasets."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from enum import Enum


class Subset(Enum):
    """Describes the Subset a DatasetItem is assigned to."""

    NONE = 0
    TRAINING = 1
    VALIDATION = 2
    TESTING = 3
    UNLABELED = 4
    PSEUDOLABELED = 5
    UNASSIGNED = 6

    def __str__(self):
        """Returns name of subset."""
        return str(self.name)

    def __repr__(self):
        """Returns name of subset."""
        return f"Subset.{self.name}"
