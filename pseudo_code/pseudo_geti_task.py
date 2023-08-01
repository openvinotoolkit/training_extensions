import os
import tempfile
from typing import Dict, List, Optional, Union

from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.model import (
    ModelEntity,
    ModelOptimizationType,
    ModelPrecision,
    OptimizationMethod,
)
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.entities.train_parameters import TrainParameters
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from otx.v2.api.core import BaseDataset


class GetiTask:
    def __init__(self, task_environment: TaskEnvironment, output_path: Optional[str] = None) -> None:
        self.task_environment = task_environment
        self.task_type = task_environment.model_template.task_type
        self.model_name = task_environment.model_template.name
        self.labels = task_environment.get_labels()

        template_file_path = task_environment.model_template.model_template_path
        self.base_dir = os.path.abspath(os.path.dirname(template_file_path))

        # Hyperparameters.
        self._work_dir_is_temp = False
        if output_path is None:
            output_path = tempfile.mkdtemp(prefix="otx-anomalib")
            self._work_dir_is_temp = True
        self.project_path: str = output_path
        self.config = self.get_config()

        # Set default model attributes.
        self.optimization_methods: List[OptimizationMethod] = []
        self.precision = [ModelPrecision.FP32]
        self.optimization_type = ModelOptimizationType.MO

        self.model = self.load_model(model=task_environment.model)


    def load_model(self, model: ModelEntity):
        raise NotImplementedError()

    def dataset_from_entity(self, dataset_entity: DatasetEntity) -> BaseDataset:
        raise NotImplementedError()

    @staticmethod
    def covert_parameter(parameters: Union[TrainParameters, InferenceParameters]) -> Dict:
        return {
            "resume": parameters.resume,
            "update_progress": parameters.update_progress,
            "save_model": parameters.save_model
        }

    @staticmethod
    def get_dataset_entity_from_result(dataset_entity: DatasetEntity, results: Dict) -> DatasetEntity:
        # TODO: Update dataset_entity here
        return dataset_entity

    def train(
        self,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        train_parameters: TrainParameters,
        seed: Optional[int] = None,
        deterministic: bool = False,
    ) -> None:
        raise NotImplementedError()

    def infer(self, dataset: DatasetEntity, inference_parameters: InferenceParameters) -> DatasetEntity:
        raise NotImplementedError()

    def evaluate(self, output_resultset: ResultSetEntity, evaluation_metric: Optional[str] = None) -> None:
        raise NotImplementedError()

    def export(
        self,
        export_type: ExportType,
        output_model: ModelEntity,
        precision: ModelPrecision = ModelPrecision.FP32,
        dump_features: bool = False,
    ) -> None:
        raise NotImplementedError()
