"""Task of OTX Segmentation."""

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
import os
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import torch
from mmcv.utils import ConfigDict

from otx.algorithms.common.configs.training_base import TrainType
from otx.algorithms.common.tasks.base_task import TRAIN_TYPE_DIR_PATH, OTXTask
from otx.algorithms.common.utils.callback import (
    InferenceProgressCallback,
    TrainingProgressCallback,
)
from otx.algorithms.common.utils.ir import embed_ir_model_data
from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.segmentation.adapters.openvino.model_wrappers.blur import (
    get_activation_map,
)
from otx.algorithms.segmentation.configs.base import SegmentationConfig
from otx.algorithms.segmentation.utils.metadata import get_seg_model_api_configuration
from otx.api.configuration import cfg_helper
from otx.api.configuration.helper.utils import ids_to_strings
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.explain_parameters import ExplainParameters
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.inference_parameters import (
    default_progress_callback as default_infer_progress_callback,
)
from otx.api.entities.metrics import (
    CurveMetric,
    InfoMetric,
    LineChartInfo,
    MetricsGroup,
    Performance,
    ScoreMetric,
    VisualizationInfo,
    VisualizationType,
)
from otx.api.entities.model import (
    ModelEntity,
    ModelPrecision,
)
from otx.api.entities.result_media import ResultMediaEntity
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.entities.tensor import TensorEntity
from otx.api.entities.train_parameters import TrainParameters, default_progress_callback
from otx.api.serialization.label_mapper import label_schema_to_bytes
from otx.api.usecases.evaluation.metrics_helper import MetricsHelper
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from otx.api.utils.segmentation_utils import (
    create_annotation_from_segmentation_map,
    create_hard_prediction_from_soft_prediction,
)
from otx.cli.utils.multi_gpu import is_multigpu_child_process

logger = get_logger()
RECIPE_TRAIN_TYPE = {
    TrainType.Semisupervised: "semisl.py",
    TrainType.Incremental: "incremental.py",
    TrainType.Selfsupervised: "selfsl.py",
}


class OTXSegmentationTask(OTXTask, ABC):
    """Task class for OTX segmentation."""

    # pylint: disable=too-many-instance-attributes, too-many-locals
    def __init__(self, task_environment: TaskEnvironment, output_path: Optional[str] = None):
        super().__init__(task_environment, output_path)
        self._task_config = SegmentationConfig
        self._hyperparams: ConfigDict = task_environment.get_hyper_parameters(self._task_config)
        self._model_name = task_environment.model_template.name
        self._train_type = self._hyperparams.algo_backend.train_type
        self.metric = "mDice"
        self._label_dictionary = dict(enumerate(self._labels, 1))  # It should have same order as model class order

        self._model_dir = os.path.join(
            os.path.abspath(os.path.dirname(self._task_environment.model_template.model_template_path)),
            TRAIN_TYPE_DIR_PATH[self._train_type.name],
        )
        if (
            self._train_type in RECIPE_TRAIN_TYPE
            and self._train_type == TrainType.Incremental
            and self._hyperparams.learning_parameters.enable_supcon
            and not self._model_dir.endswith("supcon")
        ):
            self._model_dir = os.path.join(self._model_dir, "supcon")

        if task_environment.model is not None:
            self._load_model()

        self.data_pipeline_path = os.path.join(self._model_dir, "data_pipeline.py")

    def infer(
        self,
        dataset: DatasetEntity,
        inference_parameters: Optional[InferenceParameters] = None,
    ) -> DatasetEntity:
        """Main infer function."""
        logger.info("infer()")

        if inference_parameters is not None:
            update_progress_callback = inference_parameters.update_progress
            dump_soft_prediction = not inference_parameters.is_evaluation
            process_soft_prediction = inference_parameters.process_saliency_maps
        else:
            update_progress_callback = default_infer_progress_callback
            dump_soft_prediction = True
            process_soft_prediction = False

        update_progress_callback = default_progress_callback
        if inference_parameters is not None:
            update_progress_callback = inference_parameters.update_progress  # type: ignore

        self._time_monitor = InferenceProgressCallback(len(dataset), update_progress_callback)

        predictions = self._infer_model(dataset, InferenceParameters(is_evaluation=True))
        prediction_results = zip(predictions["eval_predictions"], predictions["feature_vectors"])
        self._add_predictions_to_dataset(prediction_results, dataset, dump_soft_prediction, process_soft_prediction)

        logger.info("Inference completed")
        return dataset

    def train(
        self,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        train_parameters: Optional[TrainParameters] = None,
        seed: Optional[int] = None,
        deterministic: bool = False,
    ):
        """Train function for OTX segmentation task.

        Actual training is processed by _train_model fucntion
        """
        logger.info("train()")
        # Check for stop signal when training has stopped.
        # If should_stop is true, training was cancelled and no new
        if self._should_stop:  # type: ignore
            logger.info("Training cancelled.")
            self._should_stop = False
            self._is_training = False
            return
        self.seed = seed
        self.deterministic = deterministic

        # Set OTX LoggerHook & Time Monitor
        if train_parameters:
            update_progress_callback = train_parameters.update_progress
        else:
            update_progress_callback = default_progress_callback
        self._time_monitor = TrainingProgressCallback(update_progress_callback)

        results = self._train_model(dataset)

        # Check for stop signal when training has stopped. If should_stop is true, training was cancelled and no new
        if self._should_stop:
            logger.info("Training cancelled.")
            self._should_stop = False
            self._is_training = False
            return

        # get output model
        model_ckpt = results.get("final_ckpt")
        if model_ckpt is None:
            logger.error("cannot find final checkpoint from the results.")
            # output_model.model_status = ModelStatus.FAILED
            return
        # update checkpoint to the newly trained model
        self._model_ckpt = model_ckpt

        # get prediction on validation set
        self._is_training = False

        # Get training metrics group from learning curves
        training_metrics, best_score = self._generate_training_metrics(self._learning_curves)
        performance = Performance(
            score=ScoreMetric(value=best_score, name=self.metric),
            dashboard_metrics=training_metrics,
        )

        logger.info(f"Final model performance: {str(performance)}")
        # save resulting model
        self.save_model(output_model)
        output_model.performance = performance
        self._is_training = False
        logger.info("train done.")

    def export(
        self,
        export_type: ExportType,
        output_model: ModelEntity,
        precision: ModelPrecision = ModelPrecision.FP32,
        dump_features: bool = True,
    ):
        """Export function of OTX Task."""
        logger.info("Exporting the model")

        self._update_model_export_metadata(output_model, export_type, precision, dump_features)
        results = self._export_model(precision, export_type, dump_features)
        outputs = results.get("outputs")
        logger.debug(f"results of run_task = {outputs}")
        if outputs is None:
            raise RuntimeError(results.get("msg"))

        if export_type == ExportType.ONNX:
            onnx_file = outputs.get("onnx")
            with open(onnx_file, "rb") as f:
                output_model.set_data("model.onnx", f.read())
        else:
            bin_file = outputs.get("bin")
            xml_file = outputs.get("xml")

            ir_extra_data = get_seg_model_api_configuration(self._task_environment.label_schema, self._hyperparams)
            embed_ir_model_data(xml_file, ir_extra_data)

            with open(bin_file, "rb") as f:
                output_model.set_data("openvino.bin", f.read())
            with open(xml_file, "rb") as f:
                output_model.set_data("openvino.xml", f.read())

        output_model.set_data("label_schema.json", label_schema_to_bytes(self._task_environment.label_schema))
        logger.info("Exporting completed")

    def explain(
        self,
        dataset: DatasetEntity,
        explain_parameters: Optional[ExplainParameters] = None,
    ) -> DatasetEntity:
        """Main explain function of OTX Task."""
        raise NotImplementedError

    def evaluate(
        self,
        output_resultset: ResultSetEntity,
        evaluation_metric: Optional[str] = None,
    ):
        """Evaluate function of OTX Segmentation Task."""
        logger.info("called evaluate()")
        if evaluation_metric is not None:
            logger.warning(
                f"Requested to use {evaluation_metric} metric, " "but parameter is ignored. Use mDice instead."
            )
        metric = MetricsHelper.compute_dice_averaged_over_pixels(output_resultset)
        logger.info(f"mDice after evaluation: {metric.overall_dice.value}")
        output_resultset.performance = metric.get_performance()
        logger.info("Evaluation completed")

    def _add_predictions_to_dataset(self, prediction_results, dataset, dump_soft_prediction, process_soft_prediction):
        """Loop over dataset again to assign predictions. Convert from MMSegmentation format to OTX format."""
        for dataset_item, (prediction, feature_vector) in zip(dataset, prediction_results):
            soft_prediction = np.transpose(prediction[0], axes=(1, 2, 0))
            hard_prediction = create_hard_prediction_from_soft_prediction(
                soft_prediction=soft_prediction,
                soft_threshold=self._hyperparams.postprocessing.soft_threshold,
                blur_strength=self._hyperparams.postprocessing.blur_strength,
            )
            annotations = create_annotation_from_segmentation_map(
                hard_prediction=hard_prediction,
                soft_prediction=soft_prediction,
                label_map=self._label_dictionary,
            )
            dataset_item.append_annotations(annotations=annotations)

            if feature_vector is not None:
                active_score = TensorEntity(name="representation_vector", numpy=feature_vector.reshape(-1))
                dataset_item.append_metadata_item(active_score, model=self._task_environment.model)

            if dump_soft_prediction:
                for label_index, label in self._label_dictionary.items():
                    current_label_soft_prediction = soft_prediction[:, :, label_index]
                    if process_soft_prediction:
                        current_label_soft_prediction = get_activation_map(current_label_soft_prediction)
                    else:
                        current_label_soft_prediction = (current_label_soft_prediction * 255).astype(np.uint8)
                    result_media = ResultMediaEntity(
                        name=label.name,
                        type="soft_prediction",
                        label=label,
                        annotation_scene=dataset_item.annotation_scene,
                        roi=dataset_item.roi,
                        numpy=current_label_soft_prediction,
                    )
                    dataset_item.append_metadata_item(result_media, model=self._task_environment.model)

    def save_model(self, output_model: ModelEntity):
        """Save best model weights in SegmentationTrainTask."""

        if is_multigpu_child_process():
            return
        logger.info("called save_model")
        buffer = io.BytesIO()
        hyperparams_str = ids_to_strings(cfg_helper.convert(self._hyperparams, dict, enum_to_str=True))
        labels = {label.name: label.color.rgb_tuple for label in self._labels}
        model_ckpt = torch.load(self._model_ckpt)
        modelinfo = {
            "model": model_ckpt,
            "config": hyperparams_str,
            "labels": labels,
            "VERSION": 1,
        }

        torch.save(modelinfo, buffer)
        output_model.set_data("weights.pth", buffer.getvalue())
        output_model.set_data(
            "label_schema.json",
            label_schema_to_bytes(self._task_environment.label_schema),
        )
        output_model.precision = self._precision

    def _generate_training_metrics(self, learning_curves):
        """Get Training metrics (epochs & scores).

        Parses the mmsegmentation logs to get metrics from the latest training run
        :return output List[MetricsGroup]
        """
        output: List[MetricsGroup] = []
        # Model architecture
        architecture = InfoMetric(name="Model architecture", value=self._model_name)
        visualization_info_architecture = VisualizationInfo(
            name="Model architecture", visualisation_type=VisualizationType.TEXT
        )
        output.append(
            MetricsGroup(
                metrics=[architecture],
                visualization_info=visualization_info_architecture,
            )
        )
        # Learning curves
        best_score = -1
        for key, curve in learning_curves.items():
            metric_curve = CurveMetric(xs=curve.x, ys=curve.y, name=key)
            if key == f"val/{self.metric}":
                best_score = max(curve.y)
            visualization_info = LineChartInfo(name=key, x_axis_label="Epoch", y_axis_label=key)
            output.append(MetricsGroup(metrics=[metric_curve], visualization_info=visualization_info))

        return output, best_score

    @abstractmethod
    def _train_model(self, dataset: DatasetEntity):
        """Train model and return the results."""
        raise NotImplementedError

    @abstractmethod
    def _infer_model(
        self,
        dataset: DatasetEntity,
        inference_parameters: Optional[InferenceParameters] = None,
    ):
        """Get inference results from dataset."""
        raise NotImplementedError

    @abstractmethod
    def _export_model(self, precision: ModelPrecision, export_format: ExportType, dump_features: bool):
        """Export model and return the results."""
        raise NotImplementedError

    @abstractmethod
    def _explain_model(self, dataset: DatasetEntity, explain_parameters: Optional[ExplainParameters]):
        """Explain model and return the results."""
        raise NotImplementedError
