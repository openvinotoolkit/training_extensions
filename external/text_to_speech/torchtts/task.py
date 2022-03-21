# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import io
import logging
import os
import math
from typing import List, Optional

import tempfile
import shutil

import torch
import numpy as np

from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.train_parameters import TrainParameters, default_progress_callback
from ote_sdk.usecases.tasks.interfaces.training_interface import ITrainingTask
from ote_sdk.usecases.tasks.interfaces.inference_interface import IInferenceTask
from ote_sdk.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from ote_sdk.usecases.tasks.interfaces.unload_interface import IUnload
from ote_sdk.configuration import cfg_helper
from ote_sdk.serialization.label_mapper import label_schema_to_bytes
from ote_sdk.configuration.helper.utils import ids_to_strings
from ote_sdk.entities.model import ModelPrecision
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType, IExportTask
from ote_sdk.entities.model import ModelEntity, ModelFormat, ModelOptimizationType
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.datasets import DatasetEntity

from torchtts.integration.utils import get_default_config
from torchtts.integration.parameters import OTETextToSpeechTaskParameters
from torchtts.pipelines.pipeline_tts import PipelineTTS
from torchtts.utils import StopCallback, build_dataloader
import pytorch_lightning as pl

from torchtts.utils import export_ir, find_file
from torchtts.datasets import TTSDatasetWithSTFT


logger = logging.getLogger(__name__)


class OTETextToSpeechTask(ITrainingTask, IInferenceTask, IEvaluationTask, IExportTask, IUnload):

    task_environment: TaskEnvironment

    def __init__(self, task_environment: TaskEnvironment):
        logger.info("Loading OTETextToSpeechTask.")
        self._scratch_space = tempfile.mkdtemp(prefix="ote-tts-scratch-")
        logger.info(f"Scratch space created at {self._scratch_space}")

        self.task_environment = task_environment

        self._cfg = get_default_config()
        self._cfg.trainer.lr = self._hyperparams.learning_parameters.learning_rate
        self._cfg.trainer.max_epochs = self._hyperparams.learning_parameters.num_epochs
        self._cfg.trainer.batch_size = self._hyperparams.learning_parameters.batch_size

        self._pipeline = PipelineTTS(self._cfg)
        self.load_model(ote_model=task_environment.model)

        self._gpus = 1 if torch.cuda.device_count() else 0
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            print(os.environ['CUDA_VISIBLE_DEVICES'], torch.cuda.device_count())
        print("GPUS: ", self._gpus)

        self.stop_callback = StopCallback()
        print(self._hyperparams.learning_parameters)

    @property
    def _hyperparams(self):
        return self.task_environment.get_hyper_parameters(OTETextToSpeechTaskParameters)

    def cancel_training(self):
        """
        Called when the user wants to abort training.
        In this example, this is not implemented.

        :return: None
        """
        logger.info("Cancel training requested.")
        self.stop_callback.stop()

    def infer(self, dataset: DatasetEntity, inference_parameters: InferenceParameters) -> DatasetEntity:
        """
        Perform inference on the given dataset.

        :param dataset: Dataset entity to analyse
        :param inference_parameters: Additional parameters for inference.
            For example, when results are generated for evaluation purposes, Saliency maps can be turned off.
        :return: Dataset that also includes the classification results
        """

        valset = TTSDatasetWithSTFT(dataset_items=dataset)

        # prepare loader
        dataloader = build_dataloader(
            valset,
            batch_size=1,
            num_workers=4,
            shuffle=False
        )

        outputs = self._pipeline.predict(dataloader)

        for dataset_item, prediction in zip(dataset, outputs):
            dataset_item.annotation_scene.append_annotations([prediction])

        return dataset

    def train_like_in_torch(self, trainset: torch.utils.data.Dataset, valset: torch.utils.data.Dataset,
              output_model: ModelEntity = None, train_parameters: Optional[TrainParameters] = None):
        """ Trains a model on a dataset """

        # prepare loader
        trainloader = build_dataloader(
            trainset,
            batch_size=self._cfg.trainer.batch_size,
            num_workers=4,
            shuffle=False,
            train=True
        )

        valloader = build_dataloader(
            valset,
            batch_size=self._cfg.trainer.batch_size,
            num_workers=4,
            shuffle=False
        )

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=self._scratch_space,
            filename="{epoch}-{loss_val:.2f}",
            save_top_k=True,
            save_last=True,
            verbose=True,
            every_n_val_epochs=1,
            monitor=self._cfg.trainer.monitor,
            mode=self._cfg.trainer.monitor_mode,
        )

        self.stop_callback.reset()
        trainer = pl.Trainer(
            gpus=self._gpus,
            max_epochs=self._cfg.trainer.max_epochs,
            accumulate_grad_batches=self._cfg.trainer.grad_batches,
            strategy=self._cfg.trainer.distributed_backend,
            val_check_interval=self._cfg.trainer.val_check_interval,
            gradient_clip_val=self._cfg.trainer.gradient_clip_val,
            precision=self._cfg.trainer.precision,
            gradient_clip_algorithm="value",
            callbacks=[checkpoint_callback, self.stop_callback],
            replace_sampler_ddp=False
        )

        trainer.fit(self._pipeline, trainloader, valloader)

        logger.info("Training finished.")

        self.save_model(output_model)

    def train(self,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        train_parameters: TrainParameters,
        )-> None:
        """ Trains a model on a dataset """

        trainset = TTSDatasetWithSTFT(dataset_items=dataset, items_type=Subset.TRAINING)
        valset = TTSDatasetWithSTFT(dataset_items=dataset, items_type=Subset.VALIDATION)

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=self._scratch_space,
            filename="{epoch}-{loss_val:.2f}",
            save_top_k=True,
            save_last=True,
            verbose=True,
            every_n_val_epochs=1,
            monitor=self._cfg.trainer.monitor,
            mode=self._cfg.trainer.monitor_mode,
        )

        self.stop_callback.reset()
        trainer = pl.Trainer(
            gpus=self._gpus,
            max_epochs=self._cfg.trainer.max_epochs,
            accumulate_grad_batches=self._cfg.trainer.grad_batches,
            strategy=self._cfg.trainer.distributed_backend,
            val_check_interval=self._cfg.trainer.val_check_interval,
            gradient_clip_val=self._cfg.trainer.gradient_clip_val,
            precision=self._cfg.trainer.precision,
            gradient_clip_algorithm="value",
            callbacks=[checkpoint_callback, self.stop_callback],
            replace_sampler_ddp=False,
        )

        self._pipeline.init_datasets(trainset, valset)

        trainer.fit(self._pipeline)

        logger.info("Training finished.")

        self.save_model(output_model)

    def evaluate(self, output_resultset: ResultSetEntity,
                 evaluation_metric: Optional[str] = None):
        l1_loss = 0.0

        for val in zip(output_resultset.prediction_dataset):
            pred = val[0].annotation_scene.annotations[0]["predict"]
            gt = val[0].annotation_scene.annotations[0]["gt"]
            l1_loss += np.mean(np.abs(pred - gt))

        if len(output_resultset.prediction_dataset):
            l1_loss = l1_loss / len(output_resultset.prediction_dataset)

        output_resultset.performance = l1_loss

        logger.info(f"Difference between generated and predicted mel-spectrogram: {l1_loss}")

    def export(self, export_type: ExportType, output_model: ModelEntity):
        assert export_type == ExportType.OPENVINO
        optimized_model_precision = ModelPrecision.FP32
        output_model.model_format = ModelFormat.OPENVINO
        output_model.optimization_type = ModelOptimizationType.MO

        with tempfile.TemporaryDirectory() as tempdir:
            optimized_model_dir = os.path.join(tempdir, "dor")
            logger.info(f'Optimized model will be temporarily saved to "{optimized_model_dir}"')
            os.makedirs(optimized_model_dir, exist_ok=True)
            try:
                model_onnx_names = self._pipeline.to_onnx(optimized_model_dir, 1)
                for name in model_onnx_names:
                    export_ir(name, optimized_model_dir=optimized_model_dir,
                              data_type=optimized_model_precision.name)

                xml_files = [os.path.join(optimized_model_dir, f) for f in os.listdir(optimized_model_dir) if f.endswith('.xml')]
                failed = True
                for xml_file in xml_files:
                    bin_file = xml_file.replace('.xml', '.bin')
                    if not os.path.exists(bin_file):
                        continue
                    failed = False

                    with open(os.path.join(optimized_model_dir, bin_file), "rb") as f:
                        output_model.set_data(os.path.basename(bin_file), f.read())
                    with open(os.path.join(optimized_model_dir, xml_file), "rb") as f:
                        output_model.set_data(os.path.basename(xml_file), f.read())
                if failed:
                    raise NameError('Error in ONNX conversion')
                output_model.precision = [optimized_model_precision]
                output_model.optimization_methods = []
            except Exception as ex:
                raise RuntimeError('Optimization was unsuccessful.') from ex
            logger.info('Exporting completed.')

    @staticmethod
    def _is_docker():
        """
        Checks whether the task runs in docker container
        :return bool: True if task runs in docker
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

    def load_model(self, ote_model: Optional[ModelEntity]):
        """
        Args:
            ote_model (Optional[ModelEntity]): OTE Model from the
                task environment.
        """

        if ote_model is None:
            logger.info("Tried to load empty model. Train from scratch")
            return

        buffer = io.BytesIO(ote_model.get_data("weights.pth"))
        model_data = torch.load(buffer, map_location=torch.device("cpu"))

        if "model" in model_data:
            model_data = model_data["model"]

        if "state_dict" in model_data:
            model_data = model_data["state_dict"]

        self._pipeline.load_state_dict(model_data)#, strict=False)

    def save_model(self, output_model: ModelEntity) -> None:
        """Save the model after training is completed.

        Args:
            output_model (ModelEntity): Output model onto which the weights are saved.
        """
        logger.info("Saving the model weights.")
        config = self._cfg
        model_info = {
            "model": self._pipeline.state_dict(),
            "config": config,
            "VERSION": 1,
        }
        buffer = io.BytesIO()
        torch.save(model_info, buffer)
        output_model.set_data("weights.pth", buffer.getvalue())
        #output_model.set_data("label_schema.json", label_schema_to_bytes(self.task_environment.label_schema))
        #self._set_metadata(output_model)
        output_model.precision = [ModelPrecision.FP32]

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
