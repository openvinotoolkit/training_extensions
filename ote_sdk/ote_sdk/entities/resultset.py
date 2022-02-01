"""This module implements the ResultSet entity"""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


import abc
import datetime
from enum import Enum
from typing import Optional

from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.id import ID
from ote_sdk.entities.metrics import NullPerformance, Performance
from ote_sdk.entities.model import ModelEntity
from ote_sdk.utils.argument_checks import check_required_and_optional_parameters_type
from ote_sdk.utils.time_utils import now


class ResultsetPurpose(Enum):
    """
    This defines the purpose of the resultset.

    EVALUATION denotes resultsets generated at Evaluation stage on validation subset.

    TEST denotes resultsets generated at Evaluation stage on test subset.

    PREEVALUATION denotes resultsets generated at Preevaluation stage (e.g., train from
    scratch) onn validation subset.
    """

    EVALUATION = 0
    TEST = 1
    PREEVALUATION = 2

    def __repr__(self):
        return str(self.name)

    def __str__(self):
        """
        Returns a user friendly representation of the ResultSetPurpose, that can be
        used for instance in a progress reporting message.
        """
        user_friendly_names = {0: "Validation", 1: "Test", 2: "Pre-validation"}
        return user_friendly_names[self.value]


class ResultSetEntity(metaclass=abc.ABCMeta):
    """
    ResultsetEntity is a class aggregating:

    - the dataset containing ground truth (based on user annotations)
    - the dataset containing predictions for the above ground truth dataset

    In addition, it links to the model which computed the predictions, as well as the performance of this model on the
    ground truth dataset.

    :param model: the model using which the prediction_dataset has been generated
    :param ground_truth_dataset: the dataset containing ground truth annotation
    :param prediction_dataset: the dataset containing prediction
    :param purpose: see :class:`ResultsetPurpose`
    :param performance: the performance of the model on the ground truth dataset
    :param creation_date: the date time which the resultset is created. Set to None to set this to
    datetime.now(datetime.timezone.utc)
    :param id: the id of the resultset. Set to ID() so that a new unique ID will be assigned upon saving.
        If the argument is None, it will be set to ID()
    """

    # pylint: disable=redefined-builtin, too-many-arguments; Requires refactor
    def __init__(
        self,
        model: ModelEntity,
        ground_truth_dataset: DatasetEntity,
        prediction_dataset: DatasetEntity,
        purpose: ResultsetPurpose = ResultsetPurpose.EVALUATION,
        performance: Optional[Performance] = None,
        creation_date: Optional[datetime.datetime] = None,
        id: Optional[ID] = None,
    ):
        # Initialization parameters validation
        check_required_and_optional_parameters_type(
            required_parameters=[
                (model, "model", ModelEntity),
                (ground_truth_dataset, "ground_truth_dataset", DatasetEntity),
                (prediction_dataset, "prediction_dataset", DatasetEntity),
                (purpose, "purpose", ResultsetPurpose),
            ],
            optional_parameters=[
                (performance, "performance", Performance),
                (creation_date, "creation_date", datetime.datetime),
                (id, "id", ID),
            ],
        )

        id = ID() if id is None else id
        performance = NullPerformance() if performance is None else performance
        creation_date = now() if creation_date is None else creation_date
        self.__id = id
        self.__model = model
        self.__prediction_dataset = prediction_dataset
        self.__ground_truth_dataset = ground_truth_dataset
        self.__performance = performance
        self.__purpose = purpose
        self.__creation_date = creation_date

    @property
    def id(self) -> ID:
        """Returns the id of the ResultSet"""
        return self.__id

    @id.setter
    def id(self, value: ID) -> None:
        self.__id = value

    @property
    def model(self) -> ModelEntity:
        """Returns the model that is used for the ResultSet"""
        return self.__model

    @model.setter
    def model(self, value: ModelEntity) -> None:
        self.__model = value

    @property
    def prediction_dataset(self) -> DatasetEntity:
        """Returns the prediction dataset that is used in the ResultSet"""
        return self.__prediction_dataset

    @prediction_dataset.setter
    def prediction_dataset(self, value: DatasetEntity) -> None:
        self.__prediction_dataset = value

    @property
    def ground_truth_dataset(self) -> DatasetEntity:
        """Returns the ground truth dataset that is used in the ResultSet"""
        return self.__ground_truth_dataset

    @ground_truth_dataset.setter
    def ground_truth_dataset(self, value: DatasetEntity) -> None:
        self.__ground_truth_dataset = value

    @property
    def performance(self) -> Performance:
        """Returns the performance of the model on the ground truth dataset"""
        return self.__performance

    @performance.setter
    def performance(self, value: Performance) -> None:
        self.__performance = value

    @property
    def purpose(self) -> ResultsetPurpose:
        """
        Returns the purpose of the ResultSet, for example ResultSetPurpose.EVALUATION
        """
        return self.__purpose

    @purpose.setter
    def purpose(self, value: ResultsetPurpose) -> None:
        self.__purpose = value

    @property
    def creation_date(self) -> datetime.datetime:
        """ "Returns the creation date of the ResultSet"""
        return self.__creation_date

    @creation_date.setter
    def creation_date(self, value: datetime.datetime) -> None:
        self.__creation_date = value

    def has_score_metric(self) -> bool:
        """
        Returns True if the resultset contains non-null performance and score value.

        :return: True if the resultset contains non-null performance and score value.
        """
        return not isinstance(self.performance, NullPerformance)

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"model={self.model}, "
            f"ground_truth_dataset={self.ground_truth_dataset}, "
            f"prediction_dataset={self.prediction_dataset}, "
            f"purpose={self.purpose}, "
            f"performance={self.performance}, "
            f"creation_date={self.creation_date}, "
            f"id={self.id})"
        )
