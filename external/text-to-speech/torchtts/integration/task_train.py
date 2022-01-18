import io
import logging
import os
import math
from typing import List, Optional
from copy import deepcopy
import tempfile
import shutil

import torch
from addict import Dict

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
from ote_sdk.entities.model import ModelEntity, ModelStatus, ModelFormat, ModelOptimizationType
from ote_sdk.entities.metadata import FloatMetadata, FloatType
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.tensor import TensorEntity
from ote_sdk.entities.result_media import ResultMediaEntity
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.subset import Subset
# default config
from torchtts.integration.utils import get_default_config
from torchtts.integration.parameters import OTETextToSpeechTaskParameters
from torchtts.pipelines.pipeline_tts import PipelineTTS
from torchtts.utils import StopCallback, build_dataloader
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers

from torchtts.utils import export_ir, find_file
from torchtts.datasets import get_tts_datasets


logger = logging.getLogger(__name__)


class OTETextToSpeechTask(ITrainingTask, IInferenceTask, IEvaluationTask, IExportTask, IUnload):

    task_environment: TaskEnvironment

    def __init__(self, task_environment: TaskEnvironment):
        logger.info("Loading OTETextToSpeechTask.")
        self._scratch_space = tempfile.mkdtemp(prefix="ote-tts-scratch-")
        logger.info(f"Scratch space created at {self._scratch_space}")

        self._task_environment = task_environment

        self._cfg = get_default_config()
        self._cfg.trainer.lr = self._hyperparams.learning_parameters.learning_rate
        self._cfg.trainer.max_epochs = self._hyperparams.learning_parameters.num_epochs
        self._cfg.trainer.batch_size = self._hyperparams.learning_parameters.batch_size

        self._pipeline = PipelineTTS(self._cfg)
        print(self._cfg)
        print(self._pipeline)

        # self._device = torch.device("cuda:0") if torch.cuda.device_count() else torch.device("cpu")
        self._gpus = 1 if torch.cuda.device_count() else 0
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            print(os.environ['CUDA_VISIBLE_DEVICES'], torch.cuda.device_count())
        print("GPUS: ", self._gpus)

        self.stop_callback = StopCallback()
        print(self._hyperparams.learning_parameters)

    @property
    def _hyperparams(self):
        return self._task_environment.get_hyper_parameters(OTETextToSpeechTaskParameters)

    def cancel_training(self):
        """
        Called when the user wants to abort training.
        In this example, this is not implemented.

        :return: None
        """
        logger.info("Cancel training requested.")
        self.stop_callback.stop()

    def save_model(self, output_model: ModelEntity):
        buffer = io.BytesIO()
        hyperparams = self._task_environment.get_hyper_parameters(OTETextToSpeechTaskParameters)
        hyperparams_str = ids_to_strings(cfg_helper.convert(hyperparams, dict, enum_to_str=True))
        modelinfo = {
            'model': self._pipeline.state_dict(),
            'config': hyperparams_str,
            'vocab': self._pipeline.tokenizer.vocab(),
            'VERSION': 1
        }
        torch.save(modelinfo, buffer)
        output_model.set_data("weights.ckpt", buffer.getvalue())

    def infer(self, data_info,
              inference_parameters: Optional[InferenceParameters] = None) -> DatasetEntity:
        """
        Perform inference on the given dataset.

        :param dataset: Dataset entity to analyse
        :param inference_parameters: Additional parameters for inference.
            For example, when results are generated for evaluation purposes, Saliency maps can be turned off.
        :return: Dataset that also includes the classification results
        """

        cfg_data = Dict({
            "test_ann_file": data_info.test_ann_file,
            "test_data_root": data_info.test_data_root,
            "training_path": "../../datasets/data_ljspeech_melgan",
            "cmudict_path": find_file(os.getcwd(), "cmu_dictionary"),
            "text_cleaners": ["english_cleaners"],
            "max_wav_value": 32768.0,
            "sampling_rate": 22050,
            "filter_length": 1024,
            "hop_length": 256,
            "win_length": 1024,
            "n_mel_channels": 80,
            "mel_fmin": 0.0,
            "mel_fmax": 8000.0,
            "add_noise": True,
            "add_blank": True
        })

        dataset = get_tts_datasets(cfg_data)

        # prepare loader
        dataloader = build_dataloader(
            dataset,
            batch_size=1,
            num_workers=4,
            shuffle=False
        )

        outputs = self._pipeline.predict(dataloader)

        return outputs

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
        if self.stop_callback.check_stop():
            logger.info('Training cancelled.')
            return

        logger.info("Training finished.")

        self.save_model(output_model)

    def train(self, data_info,
              output_model: ModelEntity = None, train_parameters: Optional[TrainParameters] = None):
        """ Trains a model on a dataset """

        cfg_data = Dict({
            "train_ann_file": data_info.train_ann_file,
            "train_data_root": data_info.train_data_root,
            "val_ann_file": data_info.val_ann_file,
            "val_data_root": data_info.val_data_root,
            "training_path": "../../datasets/data_ljspeech_melgan",
            "cmudict_path": find_file(os.getcwd(), "cmu_dictionary"),
            "text_cleaners": ["english_cleaners"],
            "max_wav_value": 32768.0,
            "sampling_rate": 22050,
            "filter_length": 1024,
            "hop_length": 256,
            "win_length": 1024,
            "n_mel_channels": 80,
            "mel_fmin": 0.0,
            "mel_fmax": 8000.0,
            "add_noise": True,
            "add_blank": True
        })

        trainset, valset = get_tts_datasets(cfg_data)

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

        self._pipeline.init_datasets(trainset, valset)

        trainer.fit(self._pipeline)

        if self.stop_callback.check_stop():
            logger.info('Training cancelled.')
            return

        logger.info("Training finished.")

        self.save_model(output_model)

    def evaluate(self, output_resultset: ResultSetEntity, evaluation_metric: Optional[str] = None):
        metrics = self._pipeline.compute_metrics(output_resultset)
        logger.info(f"Computes performance of {metrics}")

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
                output_model.model_status = ModelStatus.SUCCESS
            except Exception as ex:
                output_model.model_status = ModelStatus.FAILED
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
