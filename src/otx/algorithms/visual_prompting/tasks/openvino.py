"""OpenVINO Visual Prompting Task."""

# Copyright (C) 2023 Intel Corporation
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

import io
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from zipfile import ZipFile

import attr
import numpy as np
from openvino.model_api.adapters import OpenvinoAdapter, create_core
from openvino.model_api.models import Model

from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.common.utils.utils import get_default_async_reqs_num
from otx.algorithms.visual_prompting.adapters.openvino.model_wrappers import (
    VisualPromptingOpenvinoAdapter,
)
from otx.algorithms.visual_prompting.adapters.pytorch_lightning.datasets.dataset import (
    OTXVisualPromptingDataset,
    get_transform,
)
from otx.algorithms.visual_prompting.configs.base import VisualPromptingBaseConfig
from otx.api.entities.annotation import Annotation
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.inference_parameters import (
    InferenceParameters,
    default_progress_callback,
)
from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.model import ModelEntity
from otx.api.entities.model_template import TaskType
from otx.api.entities.optimization_parameters import OptimizationParameters
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.serialization.label_mapper import LabelSchemaMapper
from otx.api.usecases.evaluation.metrics_helper import MetricsHelper
from otx.api.usecases.exportable_code import demo
from otx.api.usecases.exportable_code.inference import BaseInferencer
from otx.api.usecases.exportable_code.prediction_to_annotation_converter import (
    VisualPromptingToAnnotationConverter,
)
from otx.api.usecases.tasks.interfaces.deployment_interface import IDeploymentTask
from otx.api.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from otx.api.usecases.tasks.interfaces.inference_interface import IInferenceTask
from otx.api.usecases.tasks.interfaces.optimization_interface import (
    IOptimizationTask,
    OptimizationType,
)

logger = get_logger()


class OpenVINOVisualPromptingInferencer(BaseInferencer):
    """Inferencer implementation for Visual Prompting using OpenVINO backend.

    This inferencer has two models, image encoder and decoder.

    Args:
        hparams (VisualPromptingBaseConfig): Hyper parameters that the model should use.
        label_schema (LabelSchemaEntity): LabelSchemaEntity that was used during model training.
        model_files (Dict[str, Union[str, Path, bytes]]): Path or bytes to model to load,
            `.xml`, `.bin` or `.onnx` file.
        weight_files (Dict[str, Union[str, Path, bytes, None]], optional): Path or bytes to weights to load,
            `.xml`, `.bin` or `.onnx` file. Defaults to None.
        device (str): Device to run inference on, such as CPU, GPU or MYRIAD. Defaults to "CPU".
        num_requests (int) : Maximum number of requests that the inferencer can make.
            Good value is the number of available cores. Defaults to 1.
    """

    def __init__(
        self,
        hparams: VisualPromptingBaseConfig,
        label_schema: LabelSchemaEntity,
        model_files: Dict[str, Union[str, Path, bytes]],
        weight_files: Optional[Dict[str, Union[str, Path, bytes, None]]] = {},
        device: str = "CPU",
        num_requests: int = 1,
    ):

        assert all(module in model_files for module in ["image_encoder", "decoder"])

        self.model = {}
        model_parameters = {"decoder": {"input_layouts": "image_embeddings:NCHW"}}
        self.configuration = {
            "decoder": {
                **attr.asdict(
                    hparams.postprocessing,
                    filter=lambda attr, value: attr.name
                    not in ["header", "description", "type", "visible_in_ui", "class_name"],
                )
            }
        }
        for name in ["image_encoder", "decoder"]:
            model_adapter = VisualPromptingOpenvinoAdapter(
                core=create_core(),
                model=model_files.get(name),
                weights_path=weight_files.get(name, None),
                model_parameters=model_parameters.get(name, {}),
                device=device,
                max_num_requests=num_requests,
                plugin_config={"PERFORMANCE_HINT": "THROUGHPUT"},
            )
            self.model[name] = Model.create_model(model_adapter, name, self.configuration.get(name, {}), preload=True)
        self.converter = VisualPromptingToAnnotationConverter()
        self.labels = label_schema.get_labels(include_empty=False)
        self.transform = get_transform()  # TODO (sungchul): insert args

    def pre_process(self, dataset_item: DatasetItemEntity) -> Dict[str, Any]:  # type: ignore
        """Pre-process function of OpenVINO Visual Prompting Inferencer for image encoder."""
        images, meta = self.model["image_encoder"].preprocess(dataset_item.numpy)
        prompts = OTXVisualPromptingDataset.get_prompts(dataset_item, self.labels)  # to be replaced
        prompts = self.model["decoder"].preprocess(prompts, meta)
        items = {**images, **meta, "prompts": prompts}
        return items

    def post_process(
        self, prediction: Dict[str, np.ndarray], metadata: Dict[str, Any]
    ) -> Tuple[List[Annotation], Any, Any]:
        """Post-process function of OpenVINO Visual Prompting Inferencer."""
        hard_prediction, soft_prediction = self.model["decoder"].postprocess(prediction, metadata)
        annotation = self.converter.convert_to_annotation(hard_prediction, metadata)
        return annotation, hard_prediction, soft_prediction

    def predict(self, dataset_item: DatasetItemEntity) -> List[Annotation]:  # type: ignore
        """Perform a prediction for a given input image."""
        # forward image encoder
        items = self.pre_process(dataset_item)
        image_embeddings = self.forward({"images": items["images"]})

        annotations: List[Annotation] = []
        hard_predictions: List[np.ndarray] = []
        soft_predictions: List[np.ndarray] = []
        for prompt in items["prompts"]:
            label = prompt.pop("label")
            prompt.update(image_embeddings)

            # forward decoder to get predicted mask
            prediction = self.forward_decoder(prompt)
            metadata = {"label": label, "original_size": prompt["orig_size"]}

            # set annotation for eval
            annotation, hard_prediction, soft_prediction = self.post_process(prediction, metadata)
            annotations.extend(annotation)
            hard_predictions.append(hard_prediction)
            soft_predictions.append(soft_prediction)
        return annotations

    def forward(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Forward function of OpenVINO Visual Prompting Inferencer."""
        return self.model["image_encoder"].infer_sync(inputs)

    def forward_decoder(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Forward function of OpenVINO Visual Prompting Inferencer."""
        return self.model["decoder"].infer_sync(inputs)

    def await_all(self) -> None:
        """Await all running infer requests if any."""
        self.model["image_encoder"].await_all()
        self.model["decoder"].await_all()


class OpenVINOVisualPromptingTask(IInferenceTask, IEvaluationTask, IOptimizationTask, IDeploymentTask):
    """Task implementation for Visual Prompting using OpenVINO backend."""

    def __init__(self, task_environment: TaskEnvironment) -> None:
        self.task_environment = task_environment
        self.model = self.task_environment.model
        self.model_name = self.task_environment.model_template.model_template_id
        self.inferencer = self.load_inferencer()

        labels = task_environment.get_labels(include_empty=False)
        self._label_dictionary = dict(enumerate(labels, 1))
        template_file_path = self.task_environment.model_template.model_template_path
        self._base_dir = os.path.abspath(os.path.dirname(template_file_path))
        self.task_type = TaskType.VISUAL_PROMPTING

    @property
    def hparams(self):
        """Hparams of OpenVINO Visual Prompting Task."""
        return self.task_environment.get_hyper_parameters(VisualPromptingBaseConfig)

    def load_inferencer(self) -> OpenVINOVisualPromptingInferencer:
        """Load OpenVINO Visual Prompting Inferencer."""
        if self.model is None:
            raise RuntimeError("load_inferencer failed, model is None")
        return OpenVINOVisualPromptingInferencer(
            self.hparams,
            self.task_environment.label_schema,
            {
                "image_encoder": self.model.get_data("visual_prompting_image_encoder.xml"),
                "decoder": self.model.get_data("visual_prompting_decoder.xml"),
            },
            {
                "image_encoder": self.model.get_data("visual_prompting_image_encoder.bin"),
                "decoder": self.model.get_data("visual_prompting_decoder.bin"),
            },
            num_requests=get_default_async_reqs_num(),
        )

    def infer(
        self,
        dataset: DatasetEntity,
        inference_parameters: Optional[InferenceParameters] = None,
    ) -> DatasetEntity:
        """Infer function of OpenVINOVisualPromptingTask.

        Currently, asynchronous execution is not supported, synchronous execution will be executed instead.
        """
        if inference_parameters is not None:
            update_progress_callback = inference_parameters.update_progress
            enable_async_inference = inference_parameters.enable_async_inference
        else:
            update_progress_callback = default_progress_callback
            enable_async_inference = True

        # FIXME (sungchul): Support async inference.
        if enable_async_inference:
            logger.warning("Asynchronous inference doesn't work, synchronous inference will be executed.")
            enable_async_inference = False
        predicted_validation_dataset = dataset.with_empty_annotations()

        def add_prediction(id: int, annotations: List[Annotation]):
            dataset_item = predicted_validation_dataset[id]
            dataset_item.append_annotations(annotations)

        total_time = 0.0
        dataset_size = len(dataset)
        for i, dataset_item in enumerate(dataset, 1):
            start_time = time.perf_counter()

            annotations = self.inferencer.predict(dataset_item)
            add_prediction(i - 1, annotations)

            end_time = time.perf_counter() - start_time
            total_time += end_time
            update_progress_callback(int(i / dataset_size * 100), None)

        self.inferencer.await_all()

        logger.info(f"Avg time per image: {total_time/len(dataset)} secs")
        logger.info(f"Total time: {total_time} secs")
        logger.info("Visual Prompting OpenVINO inference completed")

        return predicted_validation_dataset

    def evaluate(self, output_resultset: ResultSetEntity, evaluation_metric: Optional[str] = None):
        """Evaluate function of OpenVINOVisualPromptingTask."""
        logger.info("Computing mDice")
        metrics = MetricsHelper.compute_dice_averaged_over_pixels(output_resultset)
        logger.info(f"mDice after evaluation: {metrics.overall_dice.value}")

        output_resultset.performance = metrics.get_performance()

    def deploy(self, output_model: ModelEntity) -> None:
        """Deploy function of OpenVINOVisualPromptingTask."""
        logger.info("Deploying the model")
        if self.model is None:
            raise RuntimeError("deploy failed, model is None")

        work_dir = os.path.dirname(demo.__file__)
        parameters = {}
        parameters["converter_type"] = f"{self.task_type}"
        parameters["model_parameters"] = self.inferencer.configuration  # type: ignore
        parameters["model_parameters"]["labels"] = LabelSchemaMapper.forward(self.task_environment.label_schema)  # type: ignore # noqa: E501

        zip_buffer = io.BytesIO()
        with ZipFile(zip_buffer, "w") as arch:
            # model files
            arch.writestr(
                os.path.join("model", "visual_prompting_image_encoder.xml"),
                self.model.get_data("visual_prompting_image_encoder.xml"),
            )
            arch.writestr(
                os.path.join("model", "visual_prompting_image_encoder.bin"),
                self.model.get_data("visual_prompting_image_encoder.bin"),
            )
            arch.writestr(
                os.path.join("model", "visual_prompting_decoder.xml"),
                self.model.get_data("visual_prompting_decoder.xml"),
            )
            arch.writestr(
                os.path.join("model", "visual_prompting_decoder.bin"),
                self.model.get_data("visual_prompting_decoder.bin"),
            )
            arch.writestr(
                os.path.join("model", "config.json"),
                json.dumps(parameters, ensure_ascii=False, indent=4),
            )
            # model_wrappers files
            for root, _, files in os.walk(os.path.dirname(model_wrappers.__file__)):
                if "__pycache__" in root:
                    continue
                for file in files:
                    file_path = os.path.join(root, file)
                    arch.write(
                        file_path,
                        os.path.join(
                            "python",
                            "model_wrappers",
                            file_path.split("model_wrappers/")[0],
                        ),
                    )
            # other python files
            arch.write(os.path.join(work_dir, "requirements.txt"), os.path.join("python", "requirements.txt"))
            arch.write(os.path.join(work_dir, "LICENSE"), os.path.join("python", "LICENSE"))
            arch.write(os.path.join(work_dir, "demo.py"), os.path.join("python", "demo.py"))
            arch.write(os.path.join(work_dir, "README.md"), os.path.join(".", "README.md"))
        output_model.exportable_code = zip_buffer.getvalue()
        logger.info("Deploying completed")

    def optimize(
        self,
        optimization_type: OptimizationType,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        optimization_parameters: Optional[OptimizationParameters] = None,
    ):
        """Optimize function of OpenVINOVisualPromptingTask."""
        raise NotImplementedError
