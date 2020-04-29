"""
 Copyright (c) 2019 Intel Corporation
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
import math
import numbers
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
import warnings
from torch import nn
from torch.nn import init
from torch.nn.utils.rnn import PackedSequence

from nncf.registry import Registry
from .layer_utils import _NNCFModuleMixin


def dict_update(src, dst, recursive=True):
    for name, value in dst.items():
        if recursive and name in src and isinstance(value, dict):
            dict_update(src[name], value, recursive)
        else:
            src[name] = value


class NNCFConv1d(_NNCFModuleMixin, nn.Conv1d):
    @staticmethod
    def from_module(module):
        assert module.__class__.__name__ == nn.Conv1d.__name__
        nncf_conv = NNCFConv1d(
            module.in_channels, module.out_channels, module.kernel_size, module.stride,
            module.padding, module.dilation, module.groups, hasattr(module, 'bias')
        )
        dict_update(nncf_conv.__dict__, module.__dict__)
        return nncf_conv


class NNCFConv2d(_NNCFModuleMixin, nn.Conv2d):
    @staticmethod
    def from_module(module):
        assert module.__class__.__name__ == nn.Conv2d.__name__
        nncf_conv = NNCFConv2d(
            module.in_channels, module.out_channels, module.kernel_size, module.stride,
            module.padding, module.dilation, module.groups, hasattr(module, 'bias')
        )
        dict_update(nncf_conv.__dict__, module.__dict__)
        return nncf_conv


class NNCFLinear(_NNCFModuleMixin, nn.Linear):
    @staticmethod
    def from_module(module):
        assert module.__class__.__name__ == nn.Linear.__name__

        nncf_linear = NNCFLinear(module.in_features, module.out_features, hasattr(module, 'bias'))
        dict_update(nncf_linear.__dict__, module.__dict__)
        return nncf_linear


class NNCFConvTranspose2d(_NNCFModuleMixin, nn.ConvTranspose2d):
    @staticmethod
    def from_module(module):
        assert module.__class__.__name__ == nn.ConvTranspose2d.__name__
        args = [module.in_channels, module.out_channels, module.kernel_size, module.stride,
                module.padding, module.output_padding, module.groups, hasattr(module, 'bias'),
                module.dilation]
        if hasattr(module, 'padding_mode'):
            args.append(module.padding_mode)
        nncf_conv_transpose2d = NNCFConvTranspose2d(*args)
        dict_update(nncf_conv_transpose2d.__dict__, module.__dict__)
        return nncf_conv_transpose2d


class NNCFConv3d(_NNCFModuleMixin, nn.Conv3d):
    @staticmethod
    def from_module(module):
        assert module.__class__.__name__ == nn.Conv3d.__name__

        nncf_conv3d = NNCFConv3d(
            module.in_channels, module.out_channels, module.kernel_size, module.stride,
            module.padding, module.dilation, module.groups, hasattr(module, 'bias')
        )
        dict_update(nncf_conv3d.__dict__, module.__dict__)
        return nncf_conv3d


class NNCFConvTranspose3d(_NNCFModuleMixin, nn.ConvTranspose3d):
    @staticmethod
    def from_module(module):
        assert module.__class__.__name__ == nn.ConvTranspose3d.__name__
        args = [module.in_channels, module.out_channels, module.kernel_size, module.stride,
                module.padding, module.output_padding, module.groups, hasattr(module, 'bias'),
                module.dilation]
        if hasattr(module, 'padding_mode'):
            args.append(module.padding_mode)
        nncf_conv_transpose3d = NNCFConvTranspose3d(*args)
        dict_update(nncf_conv_transpose3d.__dict__, module.__dict__)
        return nncf_conv_transpose3d


NNCF_MODULES_DICT = {
    NNCFConv1d: nn.Conv1d,
    NNCFConv2d: nn.Conv2d,
    NNCFConv3d: nn.Conv3d,
    NNCFLinear: nn.Linear,
    NNCFConvTranspose2d: nn.ConvTranspose2d,
    NNCFConvTranspose3d: nn.ConvTranspose3d,
}

NNCF_MODULES_MAP = {k.__name__: v.__name__ for k, v in NNCF_MODULES_DICT.items()}
NNCF_MODULES = list(NNCF_MODULES_MAP.keys())


NNCF_CONV_MODULES_DICT = {
    NNCFConv1d: nn.Conv1d,
    NNCFConv2d: nn.Conv2d,
    NNCFConv3d: nn.Conv3d,
}
NNCF_DECONV_MODULES_DICT = {
    NNCFConvTranspose2d: nn.ConvTranspose2d,
    NNCFConvTranspose3d: nn.ConvTranspose3d,
}
NNCF_CONV_MODULES_MAP = {k.__name__: v.__name__ for k, v in NNCF_CONV_MODULES_DICT.items()}
NNCF_CONV_MODULES = list(NNCF_CONV_MODULES_MAP.keys())


class RNNCellBaseNNCF(nn.Module):
    __constants__ = ['input_size', 'hidden_size', 'bias']

    def __init__(self, input_size, hidden_size, bias, num_chunks):
        super(RNNCellBaseNNCF, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        linear_ih = nn.Linear(input_size, num_chunks * hidden_size, self.bias)
        linear_hh = nn.Linear(hidden_size, num_chunks * hidden_size, self.bias)
        self.weight_ih = linear_ih.weight
        self.weight_hh = linear_hh.weight
        self.bias_ih = linear_ih.bias
        self.bias_hh = linear_hh.bias
        self.linear_list = [linear_ih, linear_hh]
        self.reset_parameters()

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input_):
        if input_.size(1) != self.input_size:
            raise RuntimeError(
                "input_ has inconsistent input_size: got {}, expected {}".format(
                    input_.size(1), self.input_size))

    def check_forward_hidden(self, input_, hx, hidden_label=''):
        # type: (Tensor, Tensor, str) -> None
        if input_.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input_.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, input_, hidden):
        raise NotImplementedError


ITERATION_MODULES = Registry('iteration_modules')

@ITERATION_MODULES.register()
class LSTMCellForwardNNCF(nn.Module):
    def __init__(self, input_linear, hidden_linear):
        super().__init__()
        self.input_linear = input_linear
        self.hidden_linear = hidden_linear

    def forward(self, input_, hidden):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> (Tensor, Tensor)
        hx, cx = hidden
        gates = self.input_linear(input_) + self.hidden_linear(hx)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)
        return hy, cy


class LSTMCellNNCF(RNNCellBaseNNCF):
    def __init__(self, input_size=1, hidden_size=1, bias=True):
        super(LSTMCellNNCF, self).__init__(input_size, hidden_size, bias, num_chunks=4)
        self.cell = LSTMCellForwardNNCF(self.linear_list[0], self.linear_list[1])

    def forward(self, input_, hidden=None):
        self.check_forward_input(input_)
        if hidden is None:
            zeros = torch.zeros(input_.size(0), self.hidden_size, dtype=input_.dtype, device=input_.device)
            hidden = (zeros, zeros)
        self.check_forward_hidden(input_, hidden[0], '[0]')
        self.check_forward_hidden(input_, hidden[1], '[1]')

        return self.cell(input_, hidden)


@ITERATION_MODULES.register()
class StackedRNN(nn.Module):
    class StackedRNNResetPoint(nn.Module):
        """
        Intentionally wrap concat, which is called inside nested loops, as a separate module.
        It allows not to add new node to nncf graph on each iteration of the loops.
        """

        def forward(self, all_output, input_):
            input_ = torch.cat(all_output, input_.dim() - 1)
            return input_

    def __init__(self, inners, num_layers, lstm=False, dropout=0):
        super().__init__()
        self.lstm = lstm
        self.num_layers = num_layers
        self.num_directions = int(len(inners) / num_layers)
        self.inners = nn.ModuleList(inners)
        self.total_layers = self.num_layers * self.num_directions
        self.dropout = dropout

    def forward(self, input_, hidden, batch_sizes):
        next_hidden = []

        if self.lstm:
            hidden = list(zip(*hidden))

        for i in range(self.num_layers):
            all_output = []
            for j in range(self.num_directions):
                l = i * self.num_directions + j
                hy, output = self.inners[l](input_, hidden[l], batch_sizes)
                next_hidden.append(hy)
                all_output.append(output)

            input_ = self.StackedRNNResetPoint()(all_output, input_)
            if self.dropout != 0 and i < self.num_layers - 1:
                input_ = F.dropout(input_, p=self.dropout, training=self.training, inplace=False)

        if self.lstm:
            next_h, next_c = zip(*next_hidden)
            next_hidden = (
                torch.cat(next_h, 0).view(self.total_layers, *next_h[0].size()),
                torch.cat(next_c, 0).view(self.total_layers, *next_c[0].size())
            )
        else:
            next_hidden = torch.cat(next_hidden, 0).view(
                self.total_layers, *next_hidden[0].size())
        return next_hidden, input_


@ITERATION_MODULES.register()
class Recurrent(nn.Module):
    def __init__(self, cell, reverse=False):
        super().__init__()
        self.reverse = reverse
        self.cell = cell

    def forward(self, input_, hidden, batch_sizes=None):
        output = []
        steps = range(input_.size(0) - 1, -1, -1) if self.reverse else range(input_.size(0))
        for i in steps:
            hidden = self.cell(input_[i], hidden)
            output.append(hidden[0] if isinstance(hidden, tuple) else hidden)

        if self.reverse:
            output.reverse()
        output = torch.cat(output, 0).view(input_.size(0), *output[0].size())

        return hidden, output


def variable_recurrent_factory():
    def factory(cell, reverse=False):
        if reverse:
            return VariableRecurrentReverse(cell)
        return VariableRecurrent(cell)

    return factory


@ITERATION_MODULES.register()
class VariableRecurrent(nn.Module):
    def __init__(self, cell):
        super().__init__()
        self.cell = cell

    def forward(self, input_, hidden, batch_sizes):
        output = []
        input_offset = 0
        last_batch_size = batch_sizes[0]
        hiddens = []
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = (hidden,)
        for batch_size in batch_sizes:
            step_input = input_[input_offset:input_offset + batch_size]
            input_offset += batch_size

            dec = last_batch_size - batch_size
            if dec > 0:
                hiddens.append(tuple(h[-dec:] for h in hidden))
                hidden = tuple(h[:-dec] for h in hidden)
            last_batch_size = batch_size

            if flat_hidden:
                hidden = (self.cell(step_input, hidden[0]),)
            else:
                hidden = self.cell(step_input, hidden)

            output.append(hidden[0])
        hiddens.append(hidden)
        hiddens.reverse()

        hidden = tuple(torch.cat(h, 0) for h in zip(*hiddens))
        assert hidden[0].size(0) == batch_sizes[0]
        if flat_hidden:
            hidden = hidden[0]
        output = torch.cat(output, 0)
        return hidden, output


@ITERATION_MODULES.register()
class VariableRecurrentReverse(nn.Module):
    def __init__(self, cell):
        super().__init__()
        self.cell = cell

    def forward(self, input_, hidden, batch_sizes):
        output = []
        input_offset = input_.size(0)
        last_batch_size = batch_sizes[-1]
        initial_hidden = hidden
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = (hidden,)
            initial_hidden = (initial_hidden,)
        hidden = tuple(h[:batch_sizes[-1]] for h in hidden)
        for batch_size in reversed(batch_sizes):
            inc = batch_size - last_batch_size
            hidden = self.ReverseResetPoint()(batch_size, hidden, inc, initial_hidden, last_batch_size)
            last_batch_size = batch_size
            step_input = input_[input_offset - batch_size:input_offset]
            input_offset -= batch_size

            if flat_hidden:
                hidden = (self.cell(step_input, hidden[0]),)
            else:
                hidden = self.cell(step_input, hidden)
            output.append(hidden[0])

        output.reverse()
        output = torch.cat(output, 0)
        if flat_hidden:
            hidden = hidden[0]
        return hidden, output

    @ITERATION_MODULES.register()
    class ReverseResetPoint(nn.Module):
        """
        Intentionally wrap concat undef if condition as a separate module
        to prevent adding new node to nncf graph on each iteration
        """

        def forward(self, batch_size, hidden, inc, initial_hidden, last_batch_size):
            if inc > 0:
                hidden = tuple(torch.cat((h, ih[last_batch_size:batch_size]), 0)
                               for h, ih in zip(hidden, initial_hidden))
            return hidden


class NNCF_RNN(nn.Module):
    """Common class for RNN modules. Currently, LSTM is supported only"""

    def __init__(self, mode='LSTM', input_size=1, hidden_size=1, num_layers=1, batch_first=False,
                 dropout=0, bidirectional=False, bias=True):
        super(NNCF_RNN, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
            isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          "num_layers greater than 1, but got dropout={} and "
                          "num_layers={}".format(dropout, num_layers))

        if mode == 'LSTM':
            gate_size = 4 * hidden_size
            self.cell_type = LSTMCellForwardNNCF
        else:
            # elif mode == 'GRU':
            #     gate_size = 3 * hidden_size
            # elif mode == 'RNN_TANH':
            #     gate_size = hidden_size
            # elif mode == 'RNN_RELU':
            #     gate_size = hidden_size
            # else:
            raise ValueError("Unrecognized RNN mode: " + mode)

        self._all_weights = []
        self.cells = []
        for layer in range(num_layers):
            for direction in range(self.num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
                linear_ih = nn.Linear(layer_input_size, gate_size, bias)
                linear_hh = nn.Linear(hidden_size, gate_size, bias)
                self.cells.append(self.cell_type(linear_ih, linear_hh))
                params = (linear_ih.weight, linear_hh.weight, linear_ih.bias, linear_hh.bias)
                suffix = '_reverse' if direction == 1 else ''
                weight_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                if bias:
                    weight_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                weight_names = [x.format(layer, suffix) for x in weight_names]
                for name, param in zip(weight_names, params):
                    setattr(self, name, param)
                self._all_weights.append(weight_names)

        self.reset_parameters()
        self.variable_length = True
        self.rnn_impl = self.get_rnn_impl(self.variable_length, self.cells)

    def get_rnn_impl(self, variable_length, cells):
        if variable_length:
            rec_factory = variable_recurrent_factory()
        else:
            rec_factory = Recurrent
        inners = []
        for layer_idx in range(self.num_layers):
            idx = layer_idx * self.num_directions
            if self.bidirectional:
                layer_inners = [rec_factory(cells[idx]), rec_factory(cells[idx + 1], reverse=True)]
            else:
                layer_inners = [rec_factory(cells[idx]), ]
            inners.extend(layer_inners)
        return StackedRNN(inners,
                          self.num_layers,
                          (self.mode == 'LSTM'),
                          dropout=self.dropout)

    def check_forward_args(self, input_, hidden, batch_sizes):
        is_input_packed = batch_sizes is not None
        expected_input_dim = 2 if is_input_packed else 3
        if input_.dim() != expected_input_dim:
            raise RuntimeError(
                'input_ must have {} dimensions, got {}'.format(
                    expected_input_dim, input_.dim()))
        if self.input_size != input_.size(-1):
            raise RuntimeError(
                'input_.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input_.size(-1)))

        if is_input_packed:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input_.size(0) if self.batch_first else input_.size(1)

        expected_hidden_size = (mini_batch, self.hidden_size)

        def check_hidden_size(hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
            expected_size = self.num_layers * self.num_directions
            if expected_size != len(hx):
                raise RuntimeError('Expected number of hidden states {}, got {}'.format(expected_size, len(hx)))
            for element in hx:
                if tuple(element.size()) != expected_hidden_size:
                    raise RuntimeError(msg.format(expected_hidden_size, tuple(element.size())))

        if self.mode == 'LSTM':
            check_hidden_size(hidden[0], expected_hidden_size,
                              'Expected hidden[0] size {}, got {}')
            check_hidden_size(hidden[1], expected_hidden_size,
                              'Expected hidden[1] size {}, got {}')
        else:
            check_hidden_size(hidden, expected_hidden_size)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]

    @staticmethod
    def apply_permutation(tensor, permutation, dim=1):
        # type: (Tensor, Tensor, int) -> Tensor
        return tensor.index_select(dim, permutation)

    def permute_hidden(self, hx, permutation):
        # type: (Tuple[Tensor, Tensor], Optional[Tensor]) -> Tuple[Tensor, Tensor]
        if permutation is None:
            return hx
        return self.apply_permutation(hx[0], permutation), self.apply_permutation(hx[1], permutation)

    def prepare_hidden(self, hx, permutation):
        # type: (Tuple[Tuple[Tensor], Tuple[Tensor]], Optional[Tensor]) -> Tuple[Tuple[Tensor], Tuple[Tensor]]
        if permutation is None:
            return hx
        split_size = len(hx[0])
        concat_hx = torch.cat([torch.unsqueeze(t, 0) for t in hx[0]])
        concat_cx = torch.cat([torch.unsqueeze(t, 0) for t in hx[1]])
        permuted_hidden = self.apply_permutation(concat_hx, permutation), self.apply_permutation(concat_cx, permutation)
        hc = permuted_hidden[0].chunk(split_size, 0)
        cc = permuted_hidden[1].chunk(split_size, 0)
        hidden = (tuple(torch.squeeze(c, 0) for c in hc), tuple(torch.squeeze(c, 0) for c in cc))
        return hidden

    def forward(self, input_, hidden=None):
        is_packed = isinstance(input_, PackedSequence)

        sorted_indices = None
        unsorted_indices = None
        if is_packed:
            input_, batch_sizes, sorted_indices, unsorted_indices = input_
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = input_.size(0) if self.batch_first else input_.size(1)

        if hidden is None:
            num_directions = 2 if self.bidirectional else 1
            hidden = input_.new_zeros(self.num_layers * num_directions,
                                      max_batch_size, self.hidden_size,
                                      requires_grad=False)
            if self.mode == 'LSTM':
                hidden = (hidden, hidden)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hidden = self.prepare_hidden(hidden, sorted_indices)

        self.check_forward_args(input_, hidden, batch_sizes)

        is_currently_variable = batch_sizes is not None
        if self.variable_length and not is_currently_variable or not self.variable_length and is_currently_variable:
            # override rnn_impl, it's assumed that this should happen very seldom, as
            # usually there's only one mode active whether variable length, or constant ones
            self.rnn_impl = self.get_rnn_impl(is_currently_variable, self.cells)

        if self.batch_first and batch_sizes is None:
            input_ = input_.transpose(0, 1)

        hidden, output = self.rnn_impl(input_, hidden, batch_sizes)

        if self.batch_first and batch_sizes is None:
            output = output.transpose(0, 1)

        if is_packed:
            output = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)

        return output, self.permute_hidden(hidden, unsorted_indices)
