import queue
from typing import List

import numpy as np
import torch

from nncf.dynamic_graph.context import no_nncf_trace
from .layers import BaseQuantizer
from ..utils import get_flat_tensor_contents_string

from nncf.nncf_logger import logger as nncf_logger


class QuantizeRangeInitializer:
    def __init__(self, quantize_module: BaseQuantizer):
        self.quantize_module = quantize_module
        self.device = next(self.quantize_module.parameters()).device
        self.scale_shape = self.quantize_module.scale_shape

    def register_input(self, x: torch.Tensor):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def forward_hook(self, module, input_, output):
        return self.register_input(input_[0])

    def apply_init(self):
        raise NotImplementedError


def max_reduce_like(input_, ref_tensor_shape):
    numel = np.prod(ref_tensor_shape)
    if numel == 1:
        return input_.max()
    tmp_max = input_
    for dim_idx, dim in enumerate(ref_tensor_shape):
        if dim == 1:
            tmp_max, _ = torch.max(tmp_max, dim_idx, keepdim=True)
    return tmp_max


def min_reduce_like(input_, ref_tensor_shape):
    numel = np.prod(ref_tensor_shape)
    if numel == 1:
        return input_.min()
    tmp_min = input_
    for dim_idx, dim in enumerate(ref_tensor_shape):
        if dim == 1:
            tmp_min, _ = torch.min(tmp_min, dim_idx, keepdim=True)
    return tmp_min


def get_channel_count_and_dim_idx(scale_shape):
    channel_dim_idx = 0
    channel_count = 1
    if not isinstance(scale_shape, int):
        for dim_idx, dim in enumerate(scale_shape):
            if dim != 1:
                channel_dim_idx = dim_idx
                channel_count = 1
    return channel_count, channel_dim_idx


def split_into_channels(input_: np.ndarray, scale_shape) -> List[np.ndarray]:
    channel_count, channel_dim_idx = get_channel_count_and_dim_idx(scale_shape)
    channel_first_tensor = np.moveaxis(input_, channel_dim_idx, 0)
    ret_list = []
    for i in range(channel_count):
        ret_list.append(channel_first_tensor[i, ...])
    return ret_list


class MinMaxInitializer(QuantizeRangeInitializer):
    def __init__(self, quantize_module: 'BaseQuantizer', is_distributed,
                 log_module_name: str = None):
        super().__init__(quantize_module)
        self.min_values = torch.ones(self.scale_shape).to(self.device) * np.inf
        self.max_values = torch.ones(self.scale_shape).to(self.device) * (-np.inf)
        self.is_distributed = is_distributed
        self.log_module_name = log_module_name

    def register_input(self, x: torch.Tensor):
        with no_nncf_trace():
            self.min_values = torch.min(min_reduce_like(x, self.scale_shape),
                                        self.min_values)
            self.max_values = torch.max(max_reduce_like(x, self.scale_shape),
                                        self.max_values)

    def reset(self):
        self.min_values = torch.ones(self.scale_shape).to(self.device) * np.inf
        self.max_values = torch.ones(self.scale_shape).to(self.device) * (-np.inf)

    def apply_init(self):
        nncf_logger.debug("Statistics: min={} max={}".format(get_flat_tensor_contents_string(self.min_values),
                                                             get_flat_tensor_contents_string(self.max_values)))
        self.quantize_module.apply_minmax_init(self.min_values, self.max_values, self.is_distributed,
                                               self.log_module_name)


class MeanMinMaxInitializer(QuantizeRangeInitializer):
    def __init__(self, quantize_module: 'BaseQuantizer', is_distributed,
                 log_module_name: str = None):
        super().__init__(quantize_module)
        self.is_distributed = is_distributed
        self.log_module_name = log_module_name
        self.all_min_values = []
        self.all_max_values = []

    def register_input(self, x: torch.Tensor):
        with no_nncf_trace():
            self.all_min_values.append(min_reduce_like(x, self.scale_shape))
            self.all_max_values.append(max_reduce_like(x, self.scale_shape))

    def reset(self):
        self.all_min_values.clear()
        self.all_max_values.clear()

    def apply_init(self):
        min_values = torch.ones(self.scale_shape).to(self.device) * (-np.inf)
        max_values = torch.ones(self.scale_shape).to(self.device) * np.inf
        if self.all_min_values:
            stacked_min = torch.stack(self.all_min_values)
            min_values = stacked_min.mean(dim=0).view(self.scale_shape)
        if self.all_max_values:
            stacked_max = torch.stack(self.all_max_values)
            max_values = stacked_max.mean(dim=0).view(self.scale_shape)
        nncf_logger.debug("Statistics: min={} max={}".format(get_flat_tensor_contents_string(min_values),
                                                             get_flat_tensor_contents_string(max_values)))
        self.quantize_module.apply_minmax_init(min_values, max_values, self.is_distributed,
                                               self.log_module_name)


class ThreeSigmaInitializer(QuantizeRangeInitializer):
    def __init__(self, quantize_module: 'BaseQuantizer', is_distributed,
                 log_module_name: str = None):
        super().__init__(quantize_module)
        self.input_history = queue.Queue()
        self.is_distributed = is_distributed
        self.log_module_name = log_module_name

    def register_input(self, x: torch.Tensor):
        with no_nncf_trace():
            self.input_history.put(x.detach().cpu().numpy())

    def reset(self):
        self.input_history = queue.Queue()

    def apply_init(self):
        self.medians = torch.ones(self.scale_shape).to(self.device)
        self.median_absolute_deviations = torch.ones(self.scale_shape).to(self.device)

        channel_count, _ = get_channel_count_and_dim_idx(self.scale_shape)
        per_channel_history = [None for i in range(channel_count)]
        while not self.input_history.empty():
            entry = self.input_history.get()
            split = split_into_channels(entry, self.scale_shape)
            for i in range(channel_count):
                flat_channel_split = split[i].flatten()

                # For post-RELU quantizers exact zeros may prevail and lead to
                # zero mean and MAD - discard them
                flat_channel_split = flat_channel_split[flat_channel_split != 0]

                if per_channel_history[i] is None:
                    per_channel_history[i] = flat_channel_split
                else:
                    per_channel_history[i] = np.concatenate([per_channel_history[i], flat_channel_split])
        per_channel_median = [np.median(channel_hist) for channel_hist in per_channel_history]
        per_channel_mad = []
        for idx, median in enumerate(per_channel_median):
            # Constant factor depends on the distribution form - assuming normal
            per_channel_mad.append(1.4826 * np.median(abs(per_channel_history[idx] - median)))

        numpy_median = np.asarray(per_channel_median)
        numpy_mad = np.asarray(per_channel_mad)
        median_tensor = torch.from_numpy(numpy_median).to(self.device, dtype=torch.float)
        mad_tensor = torch.from_numpy(numpy_mad).to(self.device, dtype=torch.float)

        nncf_logger.debug("Statistics: median={} MAD={}".format(get_flat_tensor_contents_string(median_tensor),
                                                                get_flat_tensor_contents_string(mad_tensor)))
        self.quantize_module.apply_minmax_init(median_tensor - 3 * mad_tensor, median_tensor + 3 * mad_tensor,
                                               self.is_distributed,
                                               self.log_module_name)
