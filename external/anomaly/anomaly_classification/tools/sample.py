"""`sample.py`.

This is a sample python script showing how to train an end-to-end OTE Anomaly Classification Task.
"""

# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import argparse
import importlib
import os
import shutil
from argparse import Namespace
from typing import Any, cast

from anomaly_classification import (
    AnomalyClassificationTask,
    OpenVINOAnomalyClassificationTask,
)
from ote_anomalib.data.mvtec import OteMvtecDataset
from ote_anomalib.logging import get_logger
from ote_sdk.configuration.helper import create as create_hyper_parameters
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.entities.model import (
    ModelEntity,
    ModelOptimizationType,
    ModelPrecision,
    ModelStatus,
    OptimizationMethod,
)
from ote_sdk.entities.model_template import TargetDevice, parse_model_template
from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.train_parameters import TrainParameters
from ote_sdk.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType
from ote_sdk.usecases.tasks.interfaces.inference_interface import IInferenceTask
from ote_sdk.usecases.tasks.interfaces.optimization_interface import OptimizationType

logger = get_logger(__name__)


class OteAnomalyTask:
    """OTE Anomaly Classification Task."""

    def __init__(self, dataset_path: str, seed: int, model_template_path: str) -> None:
        """Initialize OteAnomalyTask.

        Args:
            dataset_path (str): Path to the MVTec dataset.
            seed (int): Seed to split the dataset into train/val/test splits.
            model_template_path (str): Path to model template.

        Example:
            >>> import os
            >>> os.getcwd()
            '~/ote/external/anomaly'

            If MVTec dataset is placed under the above directory, then we could run,

            >>> model_template_path = "./anomaly_classification/configs/padim/template.yaml"
            >>> dataset_path = "./datasets/MVTec"
            >>> seed = 0
            >>> task = OteAnomalyTask(
            ...     dataset_path=dataset_path, seed=seed, model_template_path=model_template_path
            ... )

            >>> task.train()
            Performance(score: 1.0, dashboard: (1 metric groups))

            >>> task.export()
            Performance(score: 0.9756097560975608, dashboard: (1 metric groups))
        """
        logger.info("Loading MVTec dataset.")
        self.dataset = OteMvtecDataset(path=dataset_path, seed=seed).generate()

        logger.info("Loading the model template.")
        self.model_template = parse_model_template(model_template_path)

        logger.info("Creating the task-environment.")
        self.task_environment = self.create_task_environment()

        logger.info("Creating the base Torch and OpenVINO tasks.")
        self.torch_task = self.create_task(task="base")
        self.torch_task = cast(AnomalyClassificationTask, self.torch_task)
        self.openvino_task: OpenVINOAnomalyClassificationTask

    def create_task_environment(self) -> TaskEnvironment:
        """Create task environment."""
        hyper_parameters = create_hyper_parameters(self.model_template.hyper_parameters.data)
        labels = self.dataset.get_labels()
        label_schema = LabelSchemaEntity.from_labels(labels)

        return TaskEnvironment(
            model_template=self.model_template,
            model=None,
            hyper_parameters=hyper_parameters,
            label_schema=label_schema,
        )

    def create_task(self, task: str) -> Any:
        """Create base torch or openvino task.

        Args:
            task (str): task type. Either base or openvino.

        Returns:
            Any: Base Torch or OpenVINO Task Class.

        Example:
            >>> self.create_task(task="base")
            <anomaly_classification.torch_task.AnomalyClassificationTask>

        """
        if self.model_template.entrypoints is not None:
            task_path = getattr(self.model_template.entrypoints, task)
        else:
            raise ValueError(f"Cannot create {task} task. `model_template.entrypoint` does not have {task}")

        module_name, class_name = task_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)(task_environment=self.task_environment)

    def train(self) -> None:
        """Train the base Torch model."""
        logger.info("Training the model.")
        output_model = ModelEntity(
            train_dataset=self.dataset,
            configuration=self.task_environment.get_model_configuration(),
            model_status=ModelStatus.NOT_READY,
        )
        self.torch_task.train(
            dataset=self.dataset,
            output_model=output_model,
            train_parameters=TrainParameters(),
        )

        logger.info("Inferring the base torch model on the validation set.")
        result_set = self.infer(self.torch_task, output_model)

        logger.info("Evaluating the base torch model on the validation set.")
        self.evaluate(self.torch_task, result_set)

    def infer(self, task: IInferenceTask, output_model: ModelEntity) -> ResultSetEntity:
        """Get the predictions using the base Torch or OpenVINO tasks and models.

        Args:
            task (IInferenceTask): Task to infer. Either torch or openvino.
            output_model (ModelEntity): Output model on which the weights are saved.

        Returns:
            ResultSetEntity: Results set containing the true and pred datasets.

        """
        ground_truth_validation_dataset = self.dataset.get_subset(Subset.VALIDATION)
        prediction_validation_dataset = task.infer(
            dataset=ground_truth_validation_dataset.with_empty_annotations(),
            inference_parameters=InferenceParameters(is_evaluation=True),
        )

        return ResultSetEntity(
            model=output_model,
            ground_truth_dataset=ground_truth_validation_dataset,
            prediction_dataset=prediction_validation_dataset,
        )

    @staticmethod
    def evaluate(task: IEvaluationTask, result_set: ResultSetEntity) -> None:
        """Evaluate the performance of the model.

        Args:
            task (IEvaluationTask): Task to evaluate the performance. Either torch or openvino.
            result_set (ResultSetEntity): Results set containing the true and pred datasets.

        """
        task.evaluate(result_set)
        logger.info(str(result_set.performance))

    def export(self) -> None:
        """Export the model via openvino."""
        logger.info("Exporting the model.")
        exported_model = ModelEntity(
            train_dataset=self.dataset,
            configuration=self.task_environment.get_model_configuration(),
            model_status=ModelStatus.NOT_READY,
        )
        self.torch_task.export(ExportType.OPENVINO, exported_model)
        self.task_environment.model = exported_model

        logger.info("Creating the OpenVINO Task.")

        self.openvino_task = self.create_task(task="openvino")
        self.openvino_task = cast(OpenVINOAnomalyClassificationTask, self.openvino_task)

        logger.info("Inferring the exported model on the validation set.")
        result_set = self.infer(task=self.openvino_task, output_model=exported_model)

        logger.info("Evaluating the exported model on the validation set.")
        self.evaluate(task=self.openvino_task, result_set=result_set)

    def optimize(self) -> None:
        """Optimize the model via POT."""
        logger.info("Running the POT optimization")
        optimized_model = ModelEntity(
            self.dataset,
            configuration=self.task_environment.get_model_configuration(),
            optimization_type=ModelOptimizationType.POT,
            optimization_methods=[OptimizationMethod.QUANTIZATION],
            optimization_objectives={},
            precision=[ModelPrecision.INT8],
            target_device=TargetDevice.CPU,
            performance_improvement={},
            model_size_reduction=1.0,
            model_status=ModelStatus.NOT_READY,
        )

        self.openvino_task.optimize(
            optimization_type=OptimizationType.POT,
            dataset=self.dataset.get_subset(Subset.TRAINING),
            output_model=optimized_model,
            _optimization_parameters=OptimizationParameters(),
        )

        logger.info("Inferring the optimised model on the validation set.")
        result_set = self.infer(task=self.openvino_task, output_model=optimized_model)

        logger.info("Evaluating the optimized model on the validation set.")
        self.evaluate(task=self.openvino_task, result_set=result_set)

    @staticmethod
    def clean_up() -> None:
        """Clean up the `results` directory used by `anomalib`."""
        results_dir = "./results"
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)


def parse_args() -> Namespace:
    """Parse CLI arguments.

    Returns:
        (Namespace): CLI arguments.

    """
    parser = argparse.ArgumentParser(
        description="Sample showcasing how to run Anomaly Classification Task using OTE SDK"
    )
    parser.add_argument("--model_template_path", default="./anomaly_classification/configs/padim/template.yaml")
    parser.add_argument("--dataset_path", default="./datasets/MVTec")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--seed", default=0)
    return parser.parse_args()


def main() -> None:
    """Run `sample.py` with given CLI arguments."""
    args = parse_args()
    path = os.path.join(args.dataset_path, args.category)

    task = OteAnomalyTask(dataset_path=path, seed=args.seed, model_template_path=args.model_template_path)

    task.train()
    task.export()
    task.optimize()
    task.clean_up()


if __name__ == "__main__":
    main()
