# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import io
import itertools
import logging
import os
import math
from typing import List, Optional
from copy import deepcopy
import tempfile
import shutil

import torch

from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.metrics import (LineMetricsGroup, CurveMetric, LineChartInfo,
                                      Performance, ScoreMetric, MetricsGroup)
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.train_parameters import TrainParameters, default_progress_callback
from ote_sdk.usecases.tasks.interfaces.training_interface import ITrainingTask
from ote_sdk.usecases.tasks.interfaces.inference_interface import IInferenceTask
from ote_sdk.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from ote_sdk.usecases.tasks.interfaces.unload_interface import IUnload
from ote_sdk.usecases.evaluation.metrics_helper import MetricsHelper
from ote_sdk.configuration import cfg_helper
from ote_sdk.configuration.helper.utils import ids_to_strings
from ote_sdk.entities.model import ModelPrecision
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType, IExportTask
# from ote_sdk.entities.model import ModelEntity, ModelStatus, ModelFormat, ModelOptimizationType
from ote_sdk.entities.model import ModelEntity, ModelFormat, ModelOptimizationType
from ote_sdk.entities.metadata import FloatMetadata, FloatType
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.tensor import TensorEntity
from ote_sdk.entities.result_media import ResultMediaEntity
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.entities.label import Domain, LabelEntity
# default config
from speech_to_text.ote.quartznet_cfg import get_quartznet_cfg
from speech_to_text.ote.parameters import OTESpeechToTextTaskParameters
# utils
import speech_to_text.ote.utils as ote_utils
# pytorch pipelines
from speech_to_text.datasets import AudioDataset
from speech_to_text.trainers import LightningQuartzNetTrainer
from speech_to_text.transforms import TextTokenizerYTTM
from speech_to_text.callbacks import StopCallback
from speech_to_text.loggers import LightningSpeechToTextLogger
import speech_to_text.utils as utils
# pytorch lightning
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers


logger = logging.getLogger(__name__)


class OTESpeechToTextTaskTrain(ITrainingTask, IInferenceTask, IEvaluationTask, IExportTask, IUnload):
    """
    Base Speech To Text Task for Training and Inference

    Args:
        task_environment (TaskEnvironment): OTE Task environment.
    """

    task_environment: TaskEnvironment

    def __init__(self, task_environment: TaskEnvironment):
        logger.info("Loading OTESpeechToTextTask.")
        self._scratch_space = tempfile.mkdtemp(prefix="ote-stt-scratch-")
        logger.info(f"Scratch space created at {self._scratch_space}")
        self._task_environment = task_environment
        # configuration
        self._cfg = get_quartznet_cfg()
        self._cfg.optimizer.learning_rate = self._hyperparams.learning_parameters.learning_rate
        self._cfg.optimizer.lr_scheduler = self._hyperparams.learning_parameters.lr_scheduler
        self._cfg.optimizer.warmup_steps = self._hyperparams.learning_parameters.learning_rate_warmup_steps
        self._cfg.optimizer.epochs = self._hyperparams.learning_parameters.num_epochs
        self._cfg.pipeline.batch_size = self._hyperparams.learning_parameters.batch_size
        self._cfg.pipeline.num_workers = self._hyperparams.learning_parameters.num_workers
        self._cfg.pipeline.val_check_interval = self._hyperparams.learning_parameters.val_check_interval
        self._cfg.tokenizer.vocab_size = self._hyperparams.learning_parameters.vocab_size
        self._cfg.model.params.n_mels = self._hyperparams.learning_parameters.n_mels
        # pipeline
        self._tokenizer = TextTokenizerYTTM(**self._cfg.tokenizer)
        self._pipeline = LightningQuartzNetTrainer(self._cfg, self._tokenizer)
        # self._gpus = torch.cuda.device_count()
        self._gpus = 1 if torch.cuda.device_count() else 0
        self.stop_callback = StopCallback()
        self.pl_logger = LightningSpeechToTextLogger()
        if self._task_environment.model is not None:
            self.load_model(self._task_environment.model)

    @property
    def _hyperparams(self):
        """Hyperparams"""
        return self._task_environment.get_hyper_parameters(OTESpeechToTextTaskParameters)

    def cancel_training(self):
        """
        Called when the user wants to abort training.
        """
        logger.info("Cancel training requested.")
        self.stop_callback.stop()

    def load_model(self, input_model: ModelEntity):
        """
        Load pytorch model from ModelEntity.

        Arguments:
            input_model (ModelEntity): input model.
        """
        if "weights.ckpt" in input_model.model_adapters:
            path = os.path.join(self._scratch_space, "weights.ckpt")
            with open(path, "wb") as f:
                f.write(input_model.get_data("weights.ckpt"))
            ckpt = torch.load(path, map_location="cpu")
            self._pipeline.load_state_dict(ckpt["model"])
            with open(os.path.join(self._scratch_space, "tokenizer.bpe"), "wb") as f:
                f.write(ckpt["tokenizer.bpe"])
            self._tokenizer.from_file(os.path.join(self._scratch_space, "tokenizer.bpe"))

    def save_model(self, output_model: ModelEntity):
        """
        Save pytorch model to ModelEntity.

        Arguments:
            output_model (ModelEntity): output model.
        """
        buffer = io.BytesIO()
        hyperparams = self._task_environment.get_hyper_parameters(OTESpeechToTextTaskParameters)
        hyperparams_str = ids_to_strings(cfg_helper.convert(hyperparams, dict, enum_to_str=True))
        modelinfo = {
            'model': self._pipeline.state_dict(),
            'config': hyperparams_str,
            'vocab': self._pipeline.tokenizer.state_dict(),
            'tokenizer.bpe': open(os.path.join(self._scratch_space, "tokenizer.bpe"), "rb").read(),
            'VERSION': 1
        }
        torch.save(modelinfo, buffer)
        output_model.set_data("weights.ckpt", buffer.getvalue())

    def train(
            self,
            dataset: DatasetEntity,
            output_model: ModelEntity,
            train_parameters: Optional[TrainParameters] = None
    ):
        """
        Train model on the given dataset.

        Arguments:
            dataset (DatasetEntity): Dataset entity to analyse
            output_model (ModelEntity): output model.
            train_parameters (TrainParameters): Additional parameters for training.
        """
        self._tokenizer.initialize(
            dataset = AudioDataset(
                ote_utils.ote_extract_speech_to_text_dataset(
                    dataset.get_subset(Subset.TRAINING)
                )
            ),
            model_path = os.path.join(self._scratch_space, "tokenizer.bpe"),
            tmp_file = os.path.join(self._scratch_space, "yttm_corpus.txt")
        )

        # prepare loader
        trainloader = utils.build_dataloader(
            AudioDataset(
                ote_utils.ote_extract_speech_to_text_dataset(
                    dataset.get_subset(Subset.TRAINING)
                )
            ),
            tokenizer = self._tokenizer,
            audio_transforms_cfg = self._cfg.audio_transforms.train,
            batch_size = self._cfg.pipeline.batch_size,
            num_workers = self._cfg.pipeline.num_workers,
            shuffle = True
        )
        self._pipeline.cfg.optimizer.epoch_size = len(trainloader)
        valloader = utils.build_dataloader(
            AudioDataset(
                ote_utils.ote_extract_speech_to_text_dataset(
                    dataset.get_subset(Subset.VALIDATION)
                )
            ),
            tokenizer = self._tokenizer,
            audio_transforms_cfg = self._cfg.audio_transforms.val,
            batch_size = self._cfg.pipeline.batch_size,
            num_workers = self._cfg.pipeline.num_workers,
            shuffle = False
        )

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=self._scratch_space,
            filename="{epoch}",
            save_top_k=False,
            save_last=True,
            verbose=True,
            monitor=self._cfg.pipeline.monitor,
            mode=self._cfg.pipeline.monitor_mode,
        )

        self.stop_callback.reset()
        if self._gpus > 0:
            engine = {"devices": self._gpus, "accelerator": "gpu", "strategy": "ddp"}
        else:
            engine = {"accelerator": "cpu"}
        trainer = pl.Trainer(
            max_epochs = self._cfg.optimizer.epochs,
            accumulate_grad_batches = self._cfg.pipeline.grad_batches,
            val_check_interval = self._cfg.pipeline.val_check_interval,
            gradient_clip_val = self._cfg.pipeline.gradient_clip_val,
            gradient_clip_algorithm = "value",
            callbacks=[checkpoint_callback, self.stop_callback],
            logger=self.pl_logger,
            **engine
        )

        trainer.fit(self._pipeline, trainloader, valloader)
        if self.stop_callback.check_stop():
            logger.info('Training cancelled.')
            return

        logger.info("Training finished.")

        if output_model is not None:
            self.save_model(output_model)

    def infer(
            self,
            dataset: DatasetEntity,
            inference_parameters: Optional[InferenceParameters] = None
    ) -> DatasetEntity:
        """
        Perform inference on the given dataset.

        Arguments:
            dataset (DatasetEntity): Dataset entity to analyse
            inference_parameters (InferenceParameters): Additional parameters for inference.

        Returns:
            output_dataset (DatasetEntity): Dataset that also includes the inference results.
        """

        # prepare loader
        dataloader = utils.build_dataloader(
            AudioDataset(
                ote_utils.ote_extract_speech_to_text_dataset(dataset, False),
                load_text = False,
            ),
            tokenizer = None,
            audio_transforms_cfg = self._cfg.audio_transforms.val,
            batch_size = self._cfg.pipeline.batch_size,
            num_workers = self._cfg.pipeline.num_workers,
            shuffle = False
        )

        self.stop_callback.reset()
        if self._gpus > 0:
            engine = {"devices": min(1, self._gpus), "accelerator": "gpu", "strategy": "ddp"}
        else:
            engine = {"accelerator": "cpu"}
        trainer = pl.Trainer(
            max_epochs = self._cfg.optimizer.epochs,
            accumulate_grad_batches = self._cfg.pipeline.grad_batches,
            val_check_interval = self._cfg.pipeline.val_check_interval,
            gradient_clip_val = self._cfg.pipeline.gradient_clip_val,
            gradient_clip_algorithm = "value",
            callbacks=[self.stop_callback],
            **engine
        )
        outputs = trainer.predict(self._pipeline, dataloader)
        outputs = list(itertools.chain(*outputs))

        for dataset_item, text in zip(dataset, outputs):
            dataset_item.annotation_scene.append_annotations([ote_utils.text_to_annotation(text)])

        return dataset

    def evaluate(
        self, output_resultset: ResultSetEntity, evaluation_metric: Optional[str] = None
    ):
        """
        Evaluate the performance on a result set.
        """
        print(output_resultset.prediction_dataset)
        metrics = self._pipeline.compute_metrics(
            ote_utils.ote_extract_eval_annotation(
                output_resultset.prediction_dataset,
                output_resultset.ground_truth_dataset
            )
        )
        output_resultset.performance = metrics["wer"]
        logger.info(f"Computes performance of wer: {metrics}")

    @staticmethod
    def _is_docker():
        """
        Checks whether the task runs in docker container

        ReturnsL
            bool: True if task runs in docker
        """
        path = '/proc/self/cgroup'
        is_in_docker = False
        if os.path.isfile(path):
            with open(path) as f:
                is_in_docker = is_in_docker or any('docker' in line for line in f)
        is_in_docker = is_in_docker or os.path.exists('/.dockerenv')
        return is_in_docker

    def _delete_scratch_space(self):
        """
        Remove model checkpoints and logs
        """
        if os.path.exists(self._scratch_space):
            shutil.rmtree(self._scratch_space, ignore_errors=False)

    def unload(self):
        """
        Unload the task
        """
        self._delete_scratch_space()
        if self._is_docker():
            logger.warning(
                "Got unload request. Unloading models. Throwing Segmentation Fault on purpose")
            import ctypes
            ctypes.string_at(0)
        else:
            logger.warning("Got unload request, but not on Docker. Only clearing CUDA cache")
            torch.cuda.empty_cache()
            logger.warning(f"Done unloading. "
                           f"Torch is still occupying {torch.cuda.memory_allocated()} bytes of GPU memory")

    def export(self, export_type: ExportType, output_model: ModelEntity):
        """Export model to OpenVINO IR

        Arguments:
            export_type (ExportType): Export type should be ExportType.OPENVINO
            output_model (ModelEntity): The model entity in which to write the OpenVINO IR data

        Raises:
            Exception: If export_type is not ExportType.OPENVINO
        """
        assert export_type == ExportType.OPENVINO
        optimized_model_precision = ModelPrecision.FP32
        output_model.model_format = ModelFormat.OPENVINO
        output_model.optimization_type = ModelOptimizationType.MO

        with tempfile.TemporaryDirectory() as tempdir:
            optimized_model_dir = os.path.join(tempdir, "dor")
            logger.info(f'Optimized model will be temporarily saved to "{optimized_model_dir}"')
            os.makedirs(optimized_model_dir, exist_ok=True)
            try:
                self._pipeline.export(
                    optimized_model_dir,
                    self._hyperparams.export_parameters.sequence_length,
                    to_openvino=True
                )
                bin_file = [f for f in os.listdir(optimized_model_dir) if f.endswith('.bin')][0]
                xml_file = [f for f in os.listdir(optimized_model_dir) if f.endswith('.xml')][0]
                json_file = [f for f in os.listdir(optimized_model_dir) if f.endswith('.json')][0]
                with open(os.path.join(optimized_model_dir, bin_file), "rb") as f:
                    output_model.set_data("openvino.bin", f.read())
                with open(os.path.join(optimized_model_dir, xml_file), "rb") as f:
                    output_model.set_data("openvino.xml", f.read())
                with open(os.path.join(optimized_model_dir, json_file), "rb") as f:
                    output_model.set_data("vocab.json", f.read())
                output_model.precision = [optimized_model_precision]
                output_model.optimization_methods = []
                # output_model.model_status = ModelStatus.SUCCESS
            except Exception as ex:
                # output_model.model_status = ModelStatus.FAILED
                raise RuntimeError('Optimization was unsuccessful.') from ex
            logger.info('Exporting completed.')
