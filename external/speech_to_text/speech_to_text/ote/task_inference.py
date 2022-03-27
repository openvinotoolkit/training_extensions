# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""
OpenVINO Speech To Text Task
"""

import inspect
import json
import logging
import os
import struct
import subprocess  # nosec
import sys
import tempfile
import shutil
from typing import Any, Dict, List, Optional, Union, cast
from zipfile import ZipFile

import numpy as np
from addict import Dict as ADDict
from omegaconf import ListConfig
from omegaconf.dictconfig import DictConfig
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.inference_parameters import (
    InferenceParameters,
    default_progress_callback,
)
from ote_sdk.entities.model import (
    ModelEntity,
    ModelFormat,
    ModelOptimizationType,
    ModelPrecision,
    OptimizationMethod,
)
from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.serialization.label_mapper import LabelSchemaMapper, label_schema_to_bytes

from ote_sdk.usecases.tasks.interfaces.deployment_interface import IDeploymentTask
from ote_sdk.usecases.tasks.interfaces.inference_interface import IInferenceTask
from ote_sdk.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask

import speech_to_text.ote.deploy as deploy
from speech_to_text.ote.deploy.openvino_speech_to_text import QuartzNet
from speech_to_text.ote.parameters import OTESpeechToTextTaskParameters
import speech_to_text.ote.utils as ote_utils
from speech_to_text.metrics import WordErrorRate
from tqdm.autonotebook import tqdm

logger = logging.getLogger(__name__)

class OpenVINOSpeechToTextTask(IInferenceTask, IEvaluationTask, IDeploymentTask):
    """
    OpenVINO inference task

    Args:
        task_environment (TaskEnvironment): task environment of the trained anomaly model
    """

    def __init__(self, task_environment: TaskEnvironment) -> None:
        logger.info("Loading OpenVINOSpeechToTextTask.")
        self._scratch_space = tempfile.mkdtemp(prefix="ote-stt-scratch-")
        logger.info(f"Scratch space created at {self._scratch_space}")
        self._task_environment = task_environment
        self._hparams = self._task_environment.get_hyper_parameters(OTESpeechToTextTaskParameters)
        self._inferencer = self.load_inferencer()
        self._metric = WordErrorRate()

    def env_data_file(self, name: str) -> str:
        """Get file path from environment."""
        path = os.path.join(self._scratch_space, name)
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(self._task_environment.model.get_data(name))
        return path

    def load_inferencer(self) -> QuartzNet:
        """
        Create the OpenVINO inferencer object

        Returns:
            QuartzNet object
        """
        self.env_data_file("openvino.bin")
        return QuartzNet(
            model_path = self.env_data_file("openvino.xml"),
            vocab_path = self.env_data_file("vocab.json")
        )

    def infer(self, dataset: DatasetEntity, inference_parameters: InferenceParameters) -> DatasetEntity:
        """Perform Inference.

        Args:
            dataset (DatasetEntity): Inference dataset
            inference_parameters (InferenceParameters): Inference parameters.

        Returns:
            DatasetEntity: Output dataset storing inference predictions.
        """
        if self._task_environment.model is None:
            raise Exception("task_environment.model is None. Cannot access threshold to calculate labels.")

        logger.info("Start OpenVINO inference")
        update_progress_callback = default_progress_callback
        if inference_parameters is not None:
            update_progress_callback = inference_parameters.update_progress

        for idx, dataset_item in tqdm(enumerate(dataset)):

            audio, sampling_rate = dataset_item.media.numpy
            text = self._inferencer(audio, sampling_rate)
            dataset_item.annotation_scene.append_annotations([ote_utils.text_to_annotation(text)])
            update_progress_callback(int((idx + 1) / len(dataset) * 100))
        return dataset

    def evaluate(self, output_resultset: ResultSetEntity, evaluation_metric: Optional[str] = None):
        """Evaluate the performance of the model.

        Args:
            output_resultset (ResultSetEntity): Result set storing ground truth and predicted dataset.
            evaluation_metric (Optional[str], optional): Evaluation metric. Defaults to None.
        """
        self._metric.reset()
        data = ote_utils.ote_extract_eval_annotation(
            output_resultset.prediction_dataset,
            output_resultset.ground_truth_dataset
        )
        for pred, tgt in zip(data["pred"], data["tgt"]):
            self._metric.update(pred, tgt)
        output_resultset.performance = self._metric.compute()

    def deploy(self, output_model: ModelEntity) -> None:
        """Exports the weights from ``output_model`` along with exportable code.

        Args:
            output_model (ModelEntity): Model with ``vocab.json``, ``openvino.xml`` and ``.bin`` keys

        Raises:
            Exception: If ``task_environment.model`` is None
        """
        logger.info("Deploying Model")

        if self._task_environment.model is None:
            raise Exception("task_environment.model is None. Cannot load weights.")

        work_dir = os.path.dirname(deploy.__file__)
        print(work_dir)

        with tempfile.TemporaryDirectory() as tempdir:
            shutil.copytree(work_dir, os.path.join(tempdir, "package"))
            models_dir = os.path.join(tempdir, "package", "openvino_speech_to_text", "data")
            if not os.path.exists(models_dir):
                os.mkdir(models_dir)
            for datafile in ["openvino.xml", "openvino.bin", "vocab.json"]:
                shutil.copyfile(
                    os.path.join(self.env_data_file(datafile)),
                    os.path.join(models_dir, datafile)
                )

            # create wheel package
            subprocess.run(
                [
                    sys.executable,
                    os.path.join(tempdir, "package", "setup.py"),
                    "bdist_wheel",
                    "--dist-dir",
                    tempdir,
                    "clean",
                    "--all",
                ],
                # check=True
            )
            wheel_file_name = [f for f in os.listdir(tempdir) if f.endswith(".whl")][0]

            with ZipFile(os.path.join(tempdir, "openvino.zip"), "w") as arch:
                arch.write(os.path.join(tempdir, wheel_file_name), os.path.join("python", wheel_file_name))
            with open(os.path.join(tempdir, "openvino.zip"), "rb") as output_arch:
                output_model.exportable_code = output_arch.read()
        logger.info("Deploying completed")

    def _delete_scratch_space(self):
        """
        Remove model checkpoints and logs
        """
        if os.path.exists(self._scratch_space):
            shutil.rmtree(self._scratch_space, ignore_errors=False)
