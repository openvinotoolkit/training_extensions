import os
from enum import Enum
from typing import NamedTuple

from otx.v2.api.entities.label import Domain


class TaskInfo(NamedTuple):
    """Task information.

    NamedTuple to store information about the task type like label domain, if it is
    trainable, if it is an anomaly task and if it supports global or local labels.
    """

    domain: Domain
    is_trainable: bool
    is_anomaly: bool
    is_global: bool
    is_local: bool


class TaskType(Enum):
    """The type of algorithm within the task family.

    Also contains relevant information about the task type like label domain, if it is trainable,
    if it is an anomaly task or if it supports global or local labels.

    Args:
        value (int): (Unused) Unique integer for .value property of Enum (auto() does not work)
        task_info (TaskInfo): NamedTuple containing information about the task's capabilities
    """

    def __init__(
        self,
        value: int,
        task_info: TaskInfo,
    ):
        self.domain = task_info.domain
        self.is_trainable = task_info.is_trainable
        self.is_anomaly = task_info.is_anomaly
        self.is_global = task_info.is_global
        self.is_local = task_info.is_local

    def __new__(cls, *args):
        """Returns new instance."""
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    NULL = 1, TaskInfo(
        domain=Domain.NULL,
        is_trainable=False,
        is_anomaly=False,
        is_global=False,
        is_local=False,
    )
    DATASET = 2, TaskInfo(
        domain=Domain.NULL,
        is_trainable=False,
        is_anomaly=False,
        is_global=False,
        is_local=False,
    )
    CLASSIFICATION = 3, TaskInfo(
        domain=Domain.CLASSIFICATION,
        is_trainable=True,
        is_anomaly=False,
        is_global=True,
        is_local=False,
    )
    SEGMENTATION = 4, TaskInfo(
        domain=Domain.SEGMENTATION,
        is_trainable=True,
        is_anomaly=False,
        is_global=False,
        is_local=True,
    )
    DETECTION = 5, TaskInfo(
        domain=Domain.DETECTION,
        is_trainable=True,
        is_anomaly=False,
        is_global=False,
        is_local=True,
    )
    ANOMALY_DETECTION = 6, TaskInfo(
        domain=Domain.ANOMALY_DETECTION,
        is_trainable=True,
        is_anomaly=True,
        is_global=False,
        is_local=True,
    )
    CROP = 7, TaskInfo(
        domain=Domain.NULL,
        is_trainable=False,
        is_anomaly=False,
        is_global=False,
        is_local=False,
    )
    TILE = 8, TaskInfo(
        domain=Domain.NULL,
        is_trainable=False,
        is_anomaly=False,
        is_global=False,
        is_local=False,
    )
    INSTANCE_SEGMENTATION = 9, TaskInfo(
        domain=Domain.INSTANCE_SEGMENTATION,
        is_trainable=True,
        is_anomaly=False,
        is_global=False,
        is_local=True,
    )
    ACTIVELEARNING = 10, TaskInfo(
        domain=Domain.NULL,
        is_trainable=False,
        is_anomaly=False,
        is_global=False,
        is_local=False,
    )
    ANOMALY_SEGMENTATION = 11, TaskInfo(
        domain=Domain.ANOMALY_SEGMENTATION,
        is_trainable=True,
        is_anomaly=True,
        is_global=False,
        is_local=True,
    )
    ANOMALY_CLASSIFICATION = 12, TaskInfo(
        domain=Domain.ANOMALY_CLASSIFICATION,
        is_trainable=True,
        is_anomaly=True,
        is_global=True,
        is_local=False,
    )
    ROTATED_DETECTION = 13, TaskInfo(
        domain=Domain.ROTATED_DETECTION,
        is_trainable=True,
        is_anomaly=False,
        is_global=False,
        is_local=True,
    )
    if os.getenv("FEATURE_FLAGS_OTX_ACTION_TASKS", "0") == "1":
        ACTION_CLASSIFICATION = 14, TaskInfo(
            domain=Domain.ACTION_CLASSIFICATION,
            is_trainable=True,
            is_anomaly=False,
            is_global=False,
            is_local=True,
        )
        ACTION_DETECTION = 15, TaskInfo(
            domain=Domain.ACTION_DETECTION, is_trainable=True, is_anomaly=False, is_global=False, is_local=True
        )
    VISUAL_PROMPTING = 16, TaskInfo(  # TODO: Is 16 okay when action flag is False?
        domain=Domain.VISUAL_PROMPTING,
        is_trainable=True,
        is_anomaly=False,
        is_global=False,
        is_local=True,  # TODO: check whether is it local or not
    )


class TrainType(Enum):
    """TrainType for OTX Algorithms."""

    Finetune = "Finetune"
    Semisupervised = "Semisupervised"
    Selfsupervised = "Selfsupervised"
    Incremental = "Incremental"
    Futurework = "Futurework"
