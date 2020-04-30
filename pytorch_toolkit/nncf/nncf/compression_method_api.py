#
#  Copyright (c) 2019-2020 Intel Corporation
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""
@package docstring
This package defines the API for the NNCF compression methods, so that the user could
extend the existing algorithms.
"""

import torch
from copy import copy
from functools import partial
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from nncf.config import Config
from nncf.dynamic_graph.graph_builder import create_mock_tensor
from nncf.nncf_network import NNCFNetwork
from nncf.utils import in_scope_list


class CompressionLoss(nn.Module):
    """
    Used to calculate additional loss to be added to the base loss during the
    training process. It uses the model graph to measure variables and activations
    values of the layers during the loss construction. For example, the $L_0$-based
    sparsity algorithm calculates the number of non-zero weights in convolutional
    and fully-connected layers to construct the loss function.
    """

    def forward(self):
        """
        Returns the compression loss value.
        """
        return 0

    def statistics(self):
        """
        Returns a dictionary of printable statistics.
        """
        return {}


class CompressionScheduler:
    """
    Implements the logic of compression method control during the training process.
    May change the method hyperparameters in regards to the current training step or
    epoch. For example, the sparsity method can smoothly increase the sparsity rate
    over several epochs.
    """

    def __init__(self):
        self.last_epoch = 0
        self.last_step = 0

    def step(self, last=None):
        """
        Should be called after each optimizer step during training.
        Arguments:
            `last` - specifies the initial "previous" step
        """
        if last is None:
            last = self.last_step + 1
        self.last_step = last

    def epoch_step(self, last=None):
        """
        Should be called after each training epoch.
        Arguments:
            `last` - specifies the initial "previous" epoch
        """
        if last is None:
            last = self.last_epoch + 1
        self.last_epoch = last

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def state_dict(self):
        default_keys = {'last_step', 'last_epoch'}
        return {key: val for key, val in self.__dict__.items() if key in default_keys}

    def initialize(self):
        pass


class CompressionAlgorithmController:
    """Serves as a handle to the additional modules, parameters and hooks inserted
    into the original uncompressed model in order to enable algorithm-specific compression.
    Hosts entities that are to be used during the training process, such as compression scheduler and
    compression loss."""
    def __init__(self, target_model: NNCFNetwork):
        self._model = target_model
        self._loss = CompressionLoss()
        self._scheduler = CompressionScheduler()

    @property
    def loss(self):
        return self._loss

    @property
    def scheduler(self):
        return self._scheduler

    def distributed(self):
        """
        Should be called when distributed training with multiple training processes
        is going to be used (i.e. after the model is wrapped with DistributedDataParallel).
        Any special preparations for the algorithm to properly support distributed training
        should be made inside this function.
        """

    def statistics(self):
        """
        Returns a dictionary of printable statistics.
        """
        stats = self._loss.statistics()
        if hasattr(self._model, 'statistics'):
            stats.update(self._model.statistics())
        return stats

    def export_model(self, filename, *args, **kwargs):
        """
        Used to export the compressed model for inference into the ONNX format.
        Makes method-specific preparations of the model graph,
        (e.g. removing auxiliary layers that were used for the model compression),
        then exports the model and dumps it into the output file.
        Parameters:
            `filename` - a path to the file for the exported model to be saved into.
            *args, **kwargs - if the model's `forward` requires additional parameters
            during export, specify these here.
        """
        model = self._model.eval().cpu()
        input_tensor_list = []
        for info in self._model.input_infos:
            single_batch_info = copy(info)
            input_shape = tuple([1] + list(info.shape)[1:])
            single_batch_info.shape = input_shape
            input_tensor_list.append(create_mock_tensor(single_batch_info, "cpu"))
        original_forward = model.forward
        model.forward = partial(model.forward, *args, **kwargs)
        with torch.no_grad():
            torch.onnx.export(model, tuple(input_tensor_list),
                              filename, verbose=True)
        model.forward = original_forward

    def initialize(self, data_loader: DataLoader = None, criterion: _Loss = None):
        """
        Configures certain parameters of the algorithm that could not be set during __init__
        and require access to the dataset that the model was originally trained on (for example,
        in order to do range initialization for activation quantizers) or to the
        loss function to be used during fine-tuning (for example, to determine
        quantizer precision bitwidth using HAWQ).
        """


class CompressionAlgorithmBuilder:
    """
    Determines which modifications should be made to the original FP32 model in
    order to enable algorithm-specific compression during fine-tuning. Operates
    on an NNCFNetwork object wrapping a target PyTorch model (torch.nn.Module).
    """

    def __init__(self, config: Config):
        """
        Arguments:
          `config` - a dictionary that contains parameters of compression method
        """
        self.config = config
        if not isinstance(self.config, list):
            self.ignored_scopes = self.config.get('ignored_scopes')
            self.target_scopes = self.config.get('target_scopes')

    def apply_to(self, target_model: NNCFNetwork) -> NNCFNetwork:
        """
        Applies algorithm-specific modifications to the model. Hooks to be executed during model
        forward operation may be registered using NNCFNetwork command insertion methods. Additional
        compression modules that are expected to be saved along with the network via torch.save should also be
        registered and added to the model here.
        :param target_model: An instance of NNCFNetwork for the algorithm to be applied to.
        :return: NNCFNetwork with algorithm-specific modifications applied
        """
        self._model = target_model  # type: NNCFNetwork
        return target_model

    def build_controller(self, target_model: NNCFNetwork) -> CompressionAlgorithmController:
        """
        Should be called once the compressed model target_model is fully constructed (i.e. hooks are applied and
        modules are in place. Returns a CompressionAlgorithmController object containing information
        and references to the compressed model or specific modules thereof required for the corresponding compression
        scheduler operation or compression loss calculation.
        :param target_model: An instance of NNCFNetwork with current algorithm already applied
        :return: A CompressionAlgorithmController object.
        """

    def _should_consider_scope(self, scope_str: str) -> bool:
        return (self.target_scopes is None or in_scope_list(scope_str, self.target_scopes)) \
               and not in_scope_list(scope_str, self.ignored_scopes)
