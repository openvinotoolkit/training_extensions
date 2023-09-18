"""OTX Core Engine API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from otx.v2.api.core.registry import BaseRegistry


class Engine:
    def __init__(self, work_dir: Optional[Union[str, Path]]) -> None:
        self.work_dir = work_dir
        if work_dir is not None:
            self.work_dir = Path(work_dir).resolve()
            self.work_dir.mkdir(exist_ok=True, parents=True)
        self.registry = BaseRegistry(name="base")
        self.timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")

    @abstractmethod
    def train(
        self,
        model,
        train_dataloader,
        val_dataloader=None,
        optimizer=None,
        checkpoint: Optional[Union[str, Path]] = None,
        max_iters: Optional[int] = None,
        max_epochs: Optional[int] = None,
        distributed: Optional[bool] = None,
        seed: Optional[int] = None,
        deterministic: Optional[bool] = None,
        precision: Optional[str] = None,
        val_interval: Optional[int] = None,
        **kwargs,
    ):
        r"""OTX Engine train function.

        Args:
            model : Models to be used in training.
            train_dataloader : Dataloader to be used for training.
            val_dataloader (optional): Dataloader to use for validation step in training, without which validation will not work. Defaults to None.
            optimizer (optional): The optimizer to be used for training. Defaults to None.
            checkpoint (Optional[Union[str, Path]], optional): Model checkpoint path. Defaults to None.
            max_iters (Optional[int], optional): Maximum number of iters, enter this and iteration training will proceed. Defaults to None.
            max_epochs (Optional[int], optional): Maximum number of epochs, enter this and epoch training will proceed. Defaults to None.
            distributed (Optional[bool], optional): Distributed values for distributed training. Defaults to None.
            seed (Optional[int], optional): Random seed value. Defaults to None.
            deterministic (Optional[bool], optional): Deterministic for training. Defaults to None.
            precision (Optional[str], optional): Precision of training. Defaults to None.
            val_interval (Optional[int], optional): Validation Interval for validation step. Defaults to None.

        CLI Usage:
            1. You must first prepare the dataset by referring to [dataset-preparation-document](https://openvinotoolkit.github.io/training_extensions/stable/guide/tutorials/base/how_to_train/classification.html#dataset-preparation).
            2. By default, OTX understands the task through the structure of the dataset and automatically provides model selection and training.
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
        model,
        val_dataloader,
        checkpoint: Optional[Union[str, Path]] = None,
        precision: Optional[str] = None,
        **kwargs,
    ):
        r"""OTX Engine validate function.

        Args:
            model : Models to be used in validation.
            val_dataloader :  Dataloader to use for validation.
            checkpoint (Optional[Union[str, Path]], optional): Model checkpoint path. Defaults to None.
            precision (Optional[str], optional): Precision of model. Defaults to None.

        CLI Usage:
            1. Please enter val_data_roots with the checkpoint of your model.
                ```python
                otx validate --data.val_data_roots <path_to_data_root> --checkpoint <model_checkpoint_path>
                ```
        """

    @abstractmethod
    def test(
        self,
        model,
        test_dataloader,
        checkpoint: Optional[Union[str, Path]] = None,
        precision: Optional[str] = None,
        **kwargs,
    ):
        r"""OTX Engine test function.

        Args:
            model : Models to be used in testing.
            test_dataloader : Dataloader to use for testing.
            checkpoint (Optional[Union[str, Path]], optional): Model checkpoint path. Defaults to None.
            precision (Optional[str], optional): Precision of model. Defaults to None.

        CLI Usage:
            1. Please enter test_data_roots with the checkpoint of your model.
                ```python
                otx test --data.test_data_roots <path_to_data_root> --checkpoint <model_checkpoint_path>
                ```
        """

    @abstractmethod
    def predict(
        self,
        model,
        img,
        checkpoint: Optional[Union[str, Path]] = None,
        pipeline: Optional[List[Dict]] = None,
        **kwargs,
    ) -> List[Dict]:
        r"""OTX Engine predict function.

        Args:
            model : Models to be used in prediction.
            img (optional): Image or Dataloader to use for prediction.
            checkpoint (Optional[Union[str, Path]], optional): Model checkpoint path. Defaults to None.
            pipeline (Optional[List[Dict]], optional): Data Pipeline to be used img. Defaults to None.

        Returns:
            List[Dict]: Prediction Results.

        CLI Usage:
            1. Please enter the image file want to predict and the checkpoint of model.
                ```python
                otx test --img <image_file_root> --checkpoint <model_checkpoint_path>
                ```
        """

    @abstractmethod
    def export(
        self,
        model=None,
        checkpoint: Optional[Union[str, Path]] = None,
        precision: Optional[str] = None,
        **kwargs,
    ):
        r"""OTX Engine export function.

        Args:
            model (optional): Models to be used in exporting. Defaults to None.
            checkpoint (Optional[Union[str, Path]], optional): Model checkpoint path. Defaults to None.
            precision (Optional[str], optional):Precision for exporting. Defaults to None.

        CLI Usage:
            1. Please the checkpoint of model.
                ```python
                otx export --checkpoint <model_checkpoint_path>
                ```
        """
