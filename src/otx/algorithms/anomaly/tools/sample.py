"""`sample.py`.

This is a sample python script showing how to train an end-to-end OTX Anomaly Classification Task.
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
from typing import Any, Dict, Optional, Type, Union

from otx.algorithms.anomaly.adapters.anomalib.data.dataset import (
    AnomalyClassificationDataset,
    AnomalyDetectionDataset,
    AnomalySegmentationDataset,
)
from otx.algorithms.anomaly.tasks import NNCFTask, OpenVINOTask
from otx.api.configuration.helper import create as create_hyper_parameters
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.model import ModelEntity
from otx.api.entities.model_template import TaskType, parse_model_template
from otx.api.entities.optimization_parameters import OptimizationParameters
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.entities.train_parameters import TrainParameters
from otx.api.usecases.adapters.model_adapter import ModelAdapter
from otx.api.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from otx.api.usecases.tasks.interfaces.inference_interface import IInferenceTask
from otx.api.usecases.tasks.interfaces.optimization_interface import OptimizationType
from otx.utils.logger import get_logger

logger = get_logger()


# pylint: disable=too-many-instance-attributes
class OtxAnomalyTask:
    """OTX Anomaly Classification Task."""

    def __init__(
        self,
        dataset_path: str,
        train_subset: Dict[str, str],
        val_subset: Dict[str, str],
        test_subset: Dict[str, str],
        model_template_path: str,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize OtxAnomalyTask.

        Args:
            dataset_path (str): Path to the MVTec dataset.
            train_subset (Dict[str, str]): Dictionary containing path to train annotation file and path to dataset.
            val_subset (Dict[str, str]): Dictionary containing path to validation annotation file and path to dataset.
            test_subset (Dict[str, str]): Dictionary containing path to test annotation file and path to dataset.
            model_template_path (str): Path to model template.
            seed (Optional[int]): Setting seed to a value other than 0 also marks PytorchLightning trainer's
                deterministic flag to True.

        Example:
            >>> import os
            >>> os.getcwd()
            '~/otx/external/anomaly'

            If MVTec dataset is placed under the above directory, then we could run,

            >>> model_template_path = "./configs/classification/padim/template.yaml"
            >>> dataset_path = "./datasets/MVTec"
            >>> task = OtxAnomalyTask(
            ...     dataset_path=dataset_path,
            ...     train_subset={"ann_file": train.json, "data_root": dataset_path},
            ...     val_subset={"ann_file": val.json, "data_root": dataset_path},
            ...     test_subset={"ann_file": test.json, "data_root": dataset_path},
            ...     model_template_path=model_template_path
            ... )

            >>> task.train()
            Performance(score: 1.0, dashboard: (1 metric groups))

            >>> task.export()
            Performance(score: 0.9756097560975608, dashboard: (1 metric groups))
        """
        logger.info("Loading the model template.")
        self.model_template = parse_model_template(model_template_path)

        logger.info("Loading MVTec dataset.")
        self.task_type = self.model_template.task_type

        dataclass = self.get_dataclass()

        self.dataset = dataclass(train_subset, val_subset, test_subset)

        logger.info("Creating the task-environment.")
        self.task_environment = self.create_task_environment()

        logger.info("Creating the base Torch and OpenVINO tasks.")
        self.torch_task = self.create_task(task="base")

        self.trained_model: ModelEntity
        self.openvino_task: OpenVINOTask
        self.nncf_task: NNCFTask
        self.results = {"category": dataset_path}
        self.seed = seed

    def get_dataclass(
        self,
    ) -> Union[Type[AnomalyDetectionDataset], Type[AnomalySegmentationDataset], Type[AnomalyClassificationDataset]]:
        """Gets the dataloader based on the task type.

        Raises:
            ValueError: Validates task type.

        Returns:
           Dataloader
        """
        dataclass: Union[
            Type[AnomalyDetectionDataset], Type[AnomalySegmentationDataset], Type[AnomalyClassificationDataset]
        ]
        if self.task_type == TaskType.ANOMALY_DETECTION:
            dataclass = AnomalyDetectionDataset
        elif self.task_type == TaskType.ANOMALY_SEGMENTATION:
            dataclass = AnomalySegmentationDataset
        elif self.task_type == TaskType.ANOMALY_CLASSIFICATION:
            dataclass = AnomalyClassificationDataset
        else:
            raise ValueError(f"{self.task_type} not a supported task")
        return dataclass

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

    def train(self) -> ModelEntity:
        """Train the base Torch model."""
        logger.info("Training the model.")
        output_model = ModelEntity(
            train_dataset=self.dataset,
            configuration=self.task_environment.get_model_configuration(),
        )
        self.torch_task.train(
            dataset=self.dataset, output_model=output_model, train_parameters=TrainParameters(), seed=self.seed
        )

        logger.info("Inferring the base torch model on the validation set.")
        result_set = self.infer(self.torch_task, output_model)

        logger.info("Evaluating the base torch model on the validation set.")
        self.evaluate(self.torch_task, result_set)
        self.results["torch_fp32"] = result_set.performance.score.value
        self.trained_model = output_model
        return self.trained_model

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

    def export(self) -> ModelEntity:
        """Export the model via openvino."""
        logger.info("Exporting the model.")
        exported_model = ModelEntity(
            train_dataset=self.dataset,
            configuration=self.task_environment.get_model_configuration(),
        )
        self.torch_task.export(ExportType.OPENVINO, exported_model)
        self.task_environment.model = exported_model

        logger.info("Creating the OpenVINO Task.")

        self.openvino_task = self.create_task(task="openvino")

        logger.info("Inferring the exported model on the validation set.")
        result_set = self.infer(task=self.openvino_task, output_model=exported_model)

        logger.info("Evaluating the exported model on the validation set.")
        self.evaluate(task=self.openvino_task, result_set=result_set)
        self.results["vino_fp32"] = result_set.performance.score.value

        return exported_model

    def optimize(self) -> None:
        """Optimize the model via POT."""
        logger.info("Running the POT optimization")
        optimized_model = ModelEntity(
            self.dataset,
            configuration=self.task_environment.get_model_configuration(),
        )

        self.openvino_task.optimize(
            optimization_type=OptimizationType.POT,
            dataset=self.dataset,
            output_model=optimized_model,
            optimization_parameters=OptimizationParameters(),
        )

        logger.info("Inferring the optimised model on the validation set.")
        result_set = self.infer(task=self.openvino_task, output_model=optimized_model)

        logger.info("Evaluating the optimized model on the validation set.")
        self.evaluate(task=self.openvino_task, result_set=result_set)
        self.results["pot_int8"] = result_set.performance.score.value

    def optimize_nncf(self) -> None:
        """Optimize the model via NNCF."""
        logger.info("Running the NNCF optimization")
        init_model = ModelEntity(
            self.dataset,
            configuration=self.task_environment.get_model_configuration(),
            model_adapters={"weights.pth": ModelAdapter(self.trained_model.get_data("weights.pth"))},
        )

        self.task_environment.model = init_model
        self.nncf_task = self.create_task("nncf")

        optimized_model = ModelEntity(
            self.dataset,
            configuration=self.task_environment.get_model_configuration(),
        )
        self.nncf_task.optimize(OptimizationType.NNCF, self.dataset, optimized_model)

        logger.info("Inferring the optimised model on the validation set.")
        result_set = self.infer(task=self.nncf_task, output_model=optimized_model)

        logger.info("Evaluating the optimized model on the validation set.")
        self.evaluate(task=self.nncf_task, result_set=result_set)
        self.results["torch_int8"] = result_set.performance.score.value

    def export_nncf(self) -> ModelEntity:
        """Export NNCF model via openvino."""
        logger.info("Exporting the model.")
        exported_model = ModelEntity(
            train_dataset=self.dataset,
            configuration=self.task_environment.get_model_configuration(),
        )
        self.nncf_task.export(ExportType.OPENVINO, exported_model)
        self.task_environment.model = exported_model

        logger.info("Creating the OpenVINO Task.")

        self.openvino_task = self.create_task(task="openvino")

        logger.info("Inferring the exported model on the validation set.")
        result_set = self.infer(task=self.openvino_task, output_model=exported_model)

        logger.info("Evaluating the exported model on the validation set.")
        self.evaluate(task=self.openvino_task, result_set=result_set)
        self.results["vino_int8"] = result_set.performance.score.value
        return exported_model

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
        description="Sample showcasing how to run Anomaly Classification Task using OTX SDK"
    )
    parser.add_argument(
        "--model_template_path",
        default="./configs/classification/padim/template.yaml",
    )
    parser.add_argument("--dataset_path", default="./datasets/MVTec")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--train-ann-files", required=True)
    parser.add_argument("--val-ann-files", required=True)
    parser.add_argument("--test-ann-files", required=True)
    parser.add_argument("--optimization", choices=("none", "pot", "nncf"), default="none")
    parser.add_argument("--seed", default=0)
    return parser.parse_args()


def main() -> None:
    """Run `sample.py` with given CLI arguments."""
    args = parse_args()
    path = os.path.join(args.dataset_path, args.category)

    train_subset = {"ann_file": args.train_ann_files, "data_root": path}
    val_subset = {"ann_file": args.val_ann_files, "data_root": path}
    test_subset = {"ann_file": args.test_ann_files, "data_root": path}

    task = OtxAnomalyTask(
        dataset_path=path,
        train_subset=train_subset,
        val_subset=val_subset,
        test_subset=test_subset,
        model_template_path=args.model_template_path,
        seed=args.seed,
    )

    task.train()
    task.export()

    if args.optimization == "pot":
        task.optimize()

    if args.optimization == "nncf":
        task.optimize_nncf()
        task.export_nncf()

    task.clean_up()


if __name__ == "__main__":
    main()
