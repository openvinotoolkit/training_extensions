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
import os
from subprocess import run, DEVNULL, CalledProcessError
import tempfile

import numpy as np
import onnx
from PIL import Image
import torch
from torch.onnx.symbolic_helper import parse_args
from torch.onnx.symbolic_registry import register_op

from ote.interfaces.parameters import BaseTaskParameters
from ote.interfaces.task import ITask
from ote.interfaces.dataset import IDataset

import torchreid
from torchreid import metrics
from torchreid.data.transforms import build_transforms
from torchreid.data.transforms import build_inference_transform
from torchreid.data.datasets.image import ExternalDatasetWrapper
from torchreid.models.common import ModelInterface
from torchreid.engine import build_engine
from torchreid.utils import load_pretrained_weights, set_random_seed
from scripts.default_config import (engine_run_kwargs, get_default_config,
                                    imagedata_kwargs, lr_finder_run_kwargs,
                                    lr_scheduler_kwargs, model_kwargs,
                                    optimizer_kwargs)


class ClassificationTask(ITask):
    def __init__(self, parameters: BaseTaskParameters.BaseEnvironmentParameters, num_classes: int= 2):
        self.env_parameters = parameters
        self.num_classes = num_classes
        self.monitor = None

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

        self.device = torch.device("cuda:0") if self.cfg.use_gpu else torch.device("cpu")
        self.model = self.create_model().to(self.device)

        if self.env_parameters.snapshot_path:
            with open(self.env_parameters.snapshot_path, "rb") as f:
                model_bytes = f.read()
                self.load_model_from_bytes(model_bytes)

    def train(self, train_dataset: IDataset, val_dataset: IDataset,
              parameters: BaseTaskParameters.BaseTrainingParameters=BaseTaskParameters.BaseTrainingParameters()):

        self.cfg.train.batch_size = parameters.batch_size
        self.cfg.train.lr = parameters.base_learning_rate
        self.cfg.train.max_epoch = parameters.max_num_epochs

        #train_steps = math.ceil(len(train_dataset) / self.cfg.train.batch_size)
        #validation_steps = math.ceil((len(val_dataset) / self.cfg.test.batch_size))

        set_random_seed(self.cfg.train.seed)
        self.cfg.custom_datasets.roots = [train_dataset, val_dataset]
        self.model.train()
        datamanager = torchreid.data.ImageDataManager(**imagedata_kwargs(self.cfg))
        optimizer = torchreid.optim.build_optimizer(self.model, **optimizer_kwargs(self.cfg))
        scheduler = torchreid.optim.build_lr_scheduler(optimizer, **lr_scheduler_kwargs(self.cfg))
        engine = build_engine(self.cfg, datamanager, self.model, optimizer, scheduler)
        engine.run(**engine_run_kwargs(self.cfg), tb_writer=self.monitor)

    def test(self, dataset: IDataset, parameters: BaseTaskParameters.BaseEvaluationParameters) -> (list, dict):
        self.model.eval()
        self.model.to(self.device)
        self.cfg.custom_datasets.roots = [dataset, dataset]
        datamanager = torchreid.data.ImageDataManager(**imagedata_kwargs(self.cfg))
        cmc, mAP, norm_cm = metrics.evaluate_classification(datamanager.test_loader['val']['query'],
                                                             self.model, self.cfg.use_gpu, (1, 5))
        result_metrics = {'Top-1': cmc[0], 'Top-5': cmc[1], 'mAP': mAP}

        return [], result_metrics

    def cancel(self):
        pass

    def get_training_progress(self) -> int:
        return 0

    def compress(self, parameters: BaseTaskParameters.BaseCompressParameters):
        pass

    def export(self, parameters: BaseTaskParameters.BaseExportParameters):
        transform = build_inference_transform(
            cfg.data.height,
            cfg.data.width,
            norm_mean=cfg.data.norm_mean,
            norm_std=cfg.data.norm_std,
        )
        input_img = random_image(cfg.data.height, cfg.data.width)
        input_blob = transform(input_img).unsqueeze(0)

        input_names = ['data']
        output_names = ['reid_embedding']
        dynamic_axes = {}
        register_op("group_norm", group_norm_symbolic, "", 9)

        with tempfile.TemporaryDirectory() as tmpdirname:
            onnx_model_path = os.path.join(tmpdirname, 'classification_model.onnx')
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
                    dynamic_axes=dynamic_axes,
                    opset_version=9,
                    operator_export_type=torch.onnx.OperatorExportTypes.ONNX
                )

            mean_values = str([s*255 for s in self.cfg.data.norm_mean])
            scale_values = str([s*255 for s in self.cfg.data.norm_std])

            # read yaml here to ger mean std
            command_line = f'mo.py --input_model="{onnx_model_path}" ' \
                           f'--mean_values="{mean_values}" ' \
                           f'--scale_values="{scale_values}" ' \
                           f'--output_dir="{parameters.output_folder}" ' \
                               '--reverse_input_channels'

            try:
                run('mo.py -h', stdout=DEVNULL, stderr=DEVNULL, shell=True, check=True)
            except CalledProcessError as ex:
                print('OpenVINO Model Optimizer not found, please source '
                    'openvino/bin/setupvars.sh before running this script.')
                return

            run(command_line, shell=True, check=True)

    def load_model_from_bytes(self, binary_model: bytes):
        torch_model = self.create_model()
        state_dict = torch.load(io.BytesIO(binary_model))
        load_pretrained_weights(torch_model, pretrained_dict=state_dict)
        self.model = torch_model.to(self.device)

    def get_model_bytes(self) -> bytes:
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        return buffer.getvalue()

    def create_model(self):
        model = torchreid.models.build_model(**model_kwargs(self.cfg, self.num_classes))
        return model


def random_image(height, width):
    input_size = (height, width, 3)
    img = np.random.rand(*input_size).astype(np.float32)
    img = np.uint8(img * 255)

    out_img = Image.fromarray(img)

    return out_img


@parse_args('v', 'i', 'v', 'v', 'f', 'i')
def group_norm_symbolic(g, input, num_groups, weight, bias, eps, cudnn_enabled):
    from torch.onnx.symbolic_opset9 import reshape, mul, add, reshape_as

    channels_num = input.type().sizes()[1]

    if num_groups == channels_num:
        output = g.op('InstanceNormalization', input, weight, bias, epsilon_f=eps)
    else:
        # Reshape from [n, g * cg, h, w] to [1, n * g, cg * h, w].
        x = reshape(g, input, [0, num_groups, -1, 0])
        x = reshape(g, x, [1, -1, 0, 0])
        # Normalize channel-wise.
        x = g.op('MeanVarianceNormalization', x, axes_i=[2, 3])
        # Reshape back.
        x = reshape_as(g, x, input)
        # Apply affine transform.
        x = mul(g, x, reshape(g, weight, [1, channels_num, 1, 1]))
        output = add(g, x, reshape(g, bias, [1, channels_num, 1, 1]))

    return output
