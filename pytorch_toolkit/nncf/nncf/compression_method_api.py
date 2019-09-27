#
#  Copyright (c) 2019 Intel Corporation
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
from torch import nn


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


class CompressionAlgorithm:
    """
    Represents the compression method and its logic. Should contain references
    to the `CompressionScheduler`, `CompressionLoss`, and compressing model instances
    that are used in the compression method so that they are accessible in the
    training loop.
    """

    def __init__(self, model, config, input_size,
                 dummy_forward_fn=None):
        """
        Arguments:
          `model` - an instance of the model to be compressed
          `config` - a dictionary that contains parameters of compression method
          `input_size` - a tuple of values (B, C, H, W) specifying the input tensor
          dimensions for the model
          `dummy_forward_fn` - optional, a function that takes the model as an
          argument and performs a forward pass of the model with any valid
          sample input
        """
        self.config = config
        self.input_size = input_size
        self._loss = CompressionLoss()
        self._scheduler = CompressionScheduler()
        self._model = model

    @property
    def model(self):
        return self._model

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
        Preparation for the algorithm to properly support distributed training should be
        made inside this function.
        """

    def statistics(self):
        """
        Returns a dictionary of printable statistics.
        """
        stats = self._loss.statistics()
        if hasattr(self.model, 'statistics'):
            stats.update(self.model.statistics())
        return stats

    def export_model(self, filename):
        """
        Used to export the compressed model for inference into the ONNX format.
        Makes method-specific preparations of the model graph,
        (e.g. removing auxiliary layers that were used for the model compression),
        then exports the model and dumps it into the output file.
        Parameters:
            `filename` - a path to the file for the exported model to be saved into.
        """
        model = self._model.eval().cpu()
        input_shape = tuple([1] + list(self.input_size)[1:])
        with torch.no_grad():
            torch.onnx.export(model, torch.randn(input_shape), filename, verbose=True)

    def initialize(self, data_loader=None):
        """
        Configures certain parameters of the algorithm that could not be set during __init__.
        In particular, for the quantization algorithm this method calculates per-layer activation
        statistics on training dataset in order to choose proper output range for quantization.
        """
