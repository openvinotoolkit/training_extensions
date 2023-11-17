"""OTX Core Engine API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import abstractmethod
from datetime import datetime, timezone
from pathlib import Path

from otx.v2.api.core.registry import BaseRegistry
from otx.v2.api.entities.task_type import TaskType


class Engine:
    """The base class for all OTX engines.

    This class defines the common interface for all OTX engines, including methods for training and validation.

    Example:
    >>> runner = Engine(
        work_dir="output/folder/path",
    )
    """

    def __init__(self, task: TaskType, work_dir: str | Path | None = None) -> None:
        """Initialize a new instance of the Engine class.

        Args:
            task (TaskType): Task type of engine.
            work_dir (Optional[Union[str, Path]]): The working directory for the engine.

        Returns:
            None
        """
        self.task = task
        if work_dir is not None:
            self.work_dir: Path = Path(work_dir).resolve()
            self.work_dir.mkdir(exist_ok=True, parents=True)
        self.registry = BaseRegistry(name="base")
        self.timestamp: str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    @abstractmethod
    def train(
        self,
        model: str | (dict | (list | object)) | None,
        train_dataloader: dict | object | None,
        val_dataloader: dict | object | None = None,
        optimizer: dict | object | None = None,
        checkpoint: str | Path | None = None,
        max_iters: int | None = None,
        max_epochs: int | None = None,
        distributed: bool | None = None,
        seed: int | None = None,
        deterministic: bool | None = None,
        precision: str | None = None,
        val_interval: int | None = None,
        **kwargs,
    ) -> dict:
        """Train the given model using the given data loaders and optimizer.

        Args:
            model : Models to be used in training.
            train_dataloader : Dataloader to be used for training.
            val_dataloader (optional): Dataloader to use for validation step in training,
                without which validation will not work. Defaults to None.
            optimizer (optional): The optimizer to be used for training. Defaults to None.
            checkpoint (Optional[Union[str, Path]], optional): Model checkpoint path. Defaults to None.
            max_iters (Optional[int], optional): Maximum number of iters, enter this and iteration training will proceed
                Defaults to None.
            max_epochs (Optional[int], optional): Maximum number of epochs, enter this and epoch training will proceed.
                Defaults to None.
            distributed (Optional[bool], optional): Distributed values for distributed training. Defaults to None.
            seed (Optional[int], optional): Random seed value. Defaults to None.
            deterministic (Optional[bool], optional): Deterministic for training. Defaults to None.
            precision (Optional[str], optional): Precision of training. Defaults to None.
            val_interval (Optional[int], optional): Validation Interval for validation step. Defaults to None.
            **kwargs: This allows to add arguments that can be accepted by the train function of each framework engine.

        Returns:
            dict: A dictionary containing the results of the training.

        Example:
        >>> runner.train(
            model=Model(),
            train_dataloader=Dataloader(),
            max_epochs=2,
        )
        {model: Model(), checkpoint: "output/latest/weights.pth"}

        CLI Usage:
            1. You must first prepare the dataset by referring to documentation.
            2. By default, OTX understands the task through the structure of the dataset and provides model selection.
                ```python
                otx train --data.train_data_roots <path_to_data_root>
                ```
            3. Of course, you can override the various values with commands.
                ```python
                otx train --data.train_data_roots <path_to_data_root> --max_epochs 3
                ```
            4. If you have a ready configuration file, run it like this.
                ```python
                otx train --config <config_file_path>
                ```
        """

    @abstractmethod
    def validate(
        self,
        model: str | (dict | (list | object)) | None,
        val_dataloader: dict | object | None,
        checkpoint: str | Path | None = None,
        precision: str | None = None,
        **kwargs,
    ) -> dict:
        """Validate the given model using the given data loader.

        Args:
            model : Models to be used in validation.
            val_dataloader :  Dataloader to use for validation.
            checkpoint (Optional[Union[str, Path]], optional): Model checkpoint path. Defaults to None.
            precision (Optional[str], optional): Precision of model. Defaults to None.
            **kwargs: This allows to add arguments that can be accepted by the validate function of framework engine.

        Returns:
            dict: A dictionary containing the results of the validation.

        Example:
        >>> runner.validate(
            model=Model(),
            val_dataloader=Dataloader(),
        )
        {"metric_score": 100.0}

        CLI Usage:
            1. Please enter val_data_roots with the checkpoint of your model.
                ```python
                otx validate --data.val_data_roots <path_to_data_root> --checkpoint <model_checkpoint_path>
                ```
        """

    @abstractmethod
    def test(
        self,
        model: str | (dict | (list | object)) | None,
        test_dataloader: dict | object | None,
        checkpoint: str | Path | None = None,
        precision: str | None = None,
        **kwargs,
    ) -> dict:
        """Test the given model using the given data loader.

        Args:
            model : Models to be used in testing.
            test_dataloader : Dataloader to use for testing.
            checkpoint (Optional[Union[str, Path]], optional): Model checkpoint path. Defaults to None.
            precision (Optional[str], optional): Precision of model. Defaults to None.
            **kwargs: This allows to add arguments that can be accepted by the test function of framework engine.

        Returns:
            dict: A dictionary containing the results of the testing.

        Example:
        >>> runner.test(
            model=Model(),
            test_dataloader=Dataloader(),
        )
        {"metric_score": 100.0}

        CLI Usage:
            1. Please enter test_data_roots with the checkpoint of your model.
                ```python
                otx test --data.test_data_roots <path_to_data_root> --checkpoint <model_checkpoint_path>
                ```
        """

    @abstractmethod
    def predict(
        self,
        model: str | (dict | (list | object)) | None,
        img: str | (Path | object) | None,
        checkpoint: str | Path | None = None,
    ) -> list:
        """Predict the given model using the given image or data.

        Args:
            model : Models to be used in prediction.
            img (optional): Image or Dataloader to use for prediction.
            checkpoint (Optional[Union[str, Path]], optional): Model checkpoint path. Defaults to None.

        Returns:
            list: List of prediction results.

        Example:
        >>> runner.predict(
            model=Model(),
            img="single/image/path.img",
            checkpoint="checkpoint/weights.pth",
        )
        {"pred_score": 100.0, "pred_label": 0}

        CLI Usage:
            1. Please enter the image file want to predict and the checkpoint of model.
                ```python
                otx test --img <image_file_root> --checkpoint <model_checkpoint_path>
                ```
        """

    @abstractmethod
    def export(
        self,
        model: str | (dict | (list | object)) | None = None,
        checkpoint: str | Path | None = None,
        precision: str | None = None,
    ) -> dict:
        """Export the given model as IR Model or onnx Model.

        Args:
            model (optional): Models to be used in exporting. Defaults to None.
            checkpoint (Optional[Union[str, Path]], optional): Model checkpoint path. Defaults to None.
            precision (Optional[str], optional):Precision for exporting. Defaults to None.

        Returns:
            dict: A dictionary containing the results of the exporting.

        Example:
        >>> runner.export(
            model=Model(),
            checkpoint="checkpoint/weights.pth",
        )
        {'outputs': {'bin': 'outputs/model/openvino.bin', 'xml': 'outputs/model/openvino.xml'}}

        CLI Usage:
            1. Please the checkpoint of model.
                ```python
                otx export --checkpoint <model_checkpoint_path>
                ```
        """
