"""
 Copyright (c) 2021 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""


import io
import math
import os
from subprocess import run, DEVNULL, CalledProcessError

import torch
from torch.onnx.symbolic_registry import register_op

from ote.interfaces.parameters import BaseTaskParameters
from ote.interfaces.task import ITask
from ote.interfaces.dataset import IDataset
from ote.monitors.base_monitors import DefaultPerformanceMonitor, StopCallback
from ote.interfaces.monitoring import IMetricsMonitor, IPerformanceMonitor

import torchreid
from torchreid import metrics
from torchreid.data.transforms import build_inference_transform
from torchreid.engine import build_engine
from torchreid.utils import load_pretrained_weights, set_random_seed, resume_from_checkpoint
from scripts.default_config import (engine_run_kwargs, get_default_config,
                                    imagedata_kwargs, lr_finder_run_kwargs,
                                    lr_scheduler_kwargs, model_kwargs,
                                    optimizer_kwargs)
from torchreid.ops import DataParallel
from scripts.script_utils import build_auxiliary_model, random_image, group_norm_symbolic


class ClassificationTask(ITask):
    def __init__(self, parameters: BaseTaskParameters.BaseEnvironmentParameters,
                 metrics_monitor: IMetricsMonitor = None, num_classes: int = 2):
        self.env_parameters = parameters
        self.num_classes = num_classes
        self.stop_callback = StopCallback()
        self.metrics_monitor = metrics_monitor

        self.cfg = get_default_config()
        self.cfg.merge_from_file(self.env_parameters.config_path)
        self.cfg.use_gpu = parameters.gpu_num > 0
        self.cfg.model.classification = True
        self.cfg.custom_datasets.types = ['external_classification_wrapper', 'external_classification_wrapper']
        self.cfg.custom_datasets.names = ['train', 'val']
        self.cfg.custom_datasets.roots = ['']*2
        self.cfg.data.sources = ['train']
        self.cfg.data.targets = ['val']
        self.cfg.data.save_dir = self.env_parameters.work_dir
        self.cfg.model.pretrained = not parameters.load_weights

        for i, conf in enumerate(self.cfg.mutual_learning.aux_configs):
            if self.env_parameters.work_dir not in conf:
                self.cfg.mutual_learning.aux_configs[i] = os.path.join(self.env_parameters.work_dir, conf)

        self.device = torch.device("cuda:0") if self.cfg.use_gpu else torch.device("cpu")
        self.num_devices = min(torch.cuda.device_count(), parameters.gpu_num)
        self.model = self.create_model().to(self.device)

        self.__load_snap_if_exists()

    def __load_snap_if_exists(self, omit_classes=False):
        if self.env_parameters.load_weights:
            with open(self.env_parameters.load_weights, "rb") as f:
                model_bytes = f.read()
                self.load_model_from_bytes(model_bytes, omit_classes)

    def train(self, train_dataset: IDataset, val_dataset: IDataset,
              parameters: BaseTaskParameters.BaseTrainingParameters=BaseTaskParameters.BaseTrainingParameters(),
              performance_monitor: IPerformanceMonitor = DefaultPerformanceMonitor()):

        self.cfg.train.batch_size = parameters.batch_size
        self.cfg.train.lr = parameters.base_learning_rate
        self.cfg.train.max_epoch = parameters.max_num_epochs
        self.perf_monitor = performance_monitor

        train_steps = math.ceil(len(train_dataset) / self.cfg.train.batch_size)
        validation_steps = math.ceil((len(val_dataset) / self.cfg.test.batch_size))
        performance_monitor.init(parameters.max_num_epochs, train_steps, validation_steps)

        set_random_seed(self.cfg.train.seed)
        self.cfg.custom_datasets.roots = [train_dataset, val_dataset]
        datamanager = torchreid.data.ImageDataManager(**imagedata_kwargs(self.cfg))
        self.num_classes = datamanager.num_train_pids

        self.model = self.create_model()
        self.__load_snap_if_exists(omit_classes=True)

        num_aux_models = len(self.cfg.mutual_learning.aux_configs)

        if self.cfg.use_gpu:
            main_device_ids = list(range(self.num_devices))
            extra_device_ids = [main_device_ids for _ in range(num_aux_models)]
            self.model = DataParallel(self.model, device_ids=main_device_ids, output_device=0).cuda(main_device_ids[0])
        else:
            extra_device_ids = [None for _ in range(num_aux_models)]

        optimizer = torchreid.optim.build_optimizer(self.model, **optimizer_kwargs(self.cfg))

        if self.cfg.lr_finder.enable and self.cfg.lr_finder.mode == 'automatic' and not parameters.resume_from:
            scheduler = None
        else:
            scheduler = torchreid.optim.build_lr_scheduler(optimizer, **lr_scheduler_kwargs(self.cfg))

        if parameters.resume_from:
            self.cfg.train.start_epoch = resume_from_checkpoint(
                parameters.resume_from, self.model, optimizer=optimizer, scheduler=scheduler
            )

        if self.cfg.lr_finder.enable and not parameters.resume_from:
            if num_aux_models:
                print("Mutual learning is enabled. Learning rate will be estimated for the main model only.")

            # build new engine
            engine = build_engine(self.cfg, datamanager, self.model, optimizer, scheduler)
            lr = engine.find_lr(**lr_finder_run_kwargs(self.cfg))

            print(f"Estimated learning rate: {lr}")
            if self.cfg.lr_finder.stop_after:
                print("Finding learning rate finished. Terminate the training process")
                return

            # reload random seeds, optimizer with new lr and scheduler for it
            self.cfg.train.lr = lr
            self.cfg.lr_finder.enable = False
            set_random_seed(self.cfg.train.seed)

            optimizer = torchreid.optim.build_optimizer(self.model, **optimizer_kwargs(self.cfg))
            scheduler = torchreid.optim.build_lr_scheduler(optimizer, **lr_scheduler_kwargs(self.cfg))

        if num_aux_models:
            print('Enabled mutual learning between {} models.'.format(num_aux_models + 1))
            weights = [None] * num_aux_models

            models, optimizers, schedulers = [self.model], [optimizer], [scheduler]
            for config_file, model_weights, device_ids in \
                    zip(self.cfg.mutual_learning.aux_configs, weights, extra_device_ids):
                aux_model, aux_optimizer, aux_scheduler = build_auxiliary_model(
                    config_file, self.num_classes, self.cfg.use_gpu, device_ids, model_weights
                )

                models.append(aux_model)
                optimizers.append(aux_optimizer)
                schedulers.append(aux_scheduler)
        else:
            models, optimizers, schedulers = self.model, optimizer, scheduler

        print('Building {}-engine for {}-reid'.format(self.cfg.loss.name, self.cfg.data.type))
        self.stop_callback.reset()
        engine = build_engine(self.cfg, datamanager, models, optimizers, schedulers)
        engine.run(**engine_run_kwargs(self.cfg), tb_writer=self.metrics_monitor, perf_monitor=performance_monitor,
                   stop_callback=self.stop_callback)

        self.model = self.model.module
        self.metrics_monitor.close()

    def test(self, dataset: IDataset, parameters: BaseTaskParameters.BaseEvaluationParameters) -> (list, dict):
        self.model.eval()
        if self.cfg.use_gpu:
            main_device_ids = list(range(self.num_devices))
            self.model = DataParallel(self.model, device_ids=main_device_ids, output_device=0).cuda(main_device_ids[0])

        self.cfg.custom_datasets.roots = [dataset, dataset]
        datamanager = torchreid.data.ImageDataManager(**imagedata_kwargs(self.cfg))

        cmc, mAP, _ = metrics.evaluate_classification(datamanager.test_loader['val']['query'],
                                                      self.model, self.cfg.use_gpu, (1, 5))
        result_metrics = {'Top-1': cmc[0], 'Top-5': cmc[1], 'mAP': mAP}
        for k in result_metrics:
            result_metrics[k] = round(result_metrics[k] * 100, 2)

        return [], result_metrics

    def cancel(self):
        self.stop_callback.stop()

    def get_training_progress(self) -> int:
        return self.perf_monitor.get_training_progress()

    def compress(self, parameters: BaseTaskParameters.BaseCompressParameters):
        pass

    def export(self, parameters: BaseTaskParameters.BaseExportParameters):
        if parameters.onnx or parameters.openvino:
            transform = build_inference_transform(
                self.cfg.data.height,
                self.cfg.data.width,
                norm_mean=self.cfg.data.norm_mean,
                norm_std=self.cfg.data.norm_std,
            )
            input_img = random_image(self.cfg.data.height, self.cfg.data.width)
            input_blob = transform(input_img).unsqueeze(0).to(self.device)

            input_names = ['data']
            output_names = ['reid_embedding']
            register_op("group_norm", group_norm_symbolic, "", 9)

            onnx_model_path = os.path.join(parameters.save_model_to,
                                           os.path.splitext(os.path.basename(
                                           self.env_parameters.config_path))[0] + '.onnx')
            with torch.no_grad():
                self.model.eval()
                torch.onnx.export(
                    self.model,
                    input_blob,
                    onnx_model_path,
                    verbose=False,
                    export_params=True,
                    input_names=input_names,
                    output_names=output_names,
                    opset_version=9,
                    operator_export_type=torch.onnx.OperatorExportTypes.ONNX
                )
        if parameters.openvino:
            mean_values = str([s*255 for s in self.cfg.data.norm_mean])
            scale_values = str([s*255 for s in self.cfg.data.norm_std])

            # read yaml here to ger mean std
            command_line = f'mo.py --input_model="{onnx_model_path}" ' \
                           f'--mean_values="{mean_values}" ' \
                           f'--scale_values="{scale_values}" ' \
                           f'--output_dir="{parameters.save_model_to}" '
            if parameters.openvino_input_format == 'BGR':
                command_line += '--reverse_input_channels '

            command_line += parameters.openvino_mo_args

            try:
                run('mo.py -h', stdout=DEVNULL, stderr=DEVNULL, shell=True, check=True)
            except CalledProcessError as _:
                print('OpenVINO Model Optimizer not found, please source '
                    'openvino/bin/setupvars.sh before running this script.')
                return

            run(command_line, shell=True, check=True)

    def load_model_from_bytes(self, binary_model: bytes, omit_classes: bool = False):
        model_state = torch.load(io.BytesIO(binary_model))
        if not omit_classes:
            self.num_classes = model_state['num_classes'] if isinstance(model_state['num_classes'], int) \
                    else model_state['num_classes'][0]
        torch_model = self.create_model()
        load_pretrained_weights(torch_model, pretrained_dict=model_state)
        self.model = torch_model.to(self.device)

    def get_model_bytes(self) -> bytes:
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        return buffer.getvalue()

    def create_model(self):
        model = torchreid.models.build_model(**model_kwargs(self.cfg, self.num_classes))
        return model
