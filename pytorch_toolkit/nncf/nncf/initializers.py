import logging

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class InitializeDataLoader:
    """
    This class wraps the torch.utils.data.DataLoader class,
    enabling to return a tuple containing the input tensor
    and additional kwargs required for the forward pass
    when used as an iterator.
    Custom logic to prepare the input tensor might be
    added to the __next__ method.
    """

    def __init__(self, data_loader, kwargs, device):
        self.data_loader = data_loader
        self.kwargs = kwargs
        self.device = device

    def __iter__(self):
        return self

    def __next__(self):
        batch_input, _ = next(iter(self.data_loader))
        return (batch_input.to(self.device), self.kwargs)

    @property
    def num_workers(self):
        return self.data_loader.num_workers

    @num_workers.setter
    def num_workers(self, num_workers):
        self.data_loader.num_workers = num_workers


class DataLoaderInitializer:

    def __init__(self, apply_collected_fn, num_init_steps=None):
        super().__init__()
        self.num_init_steps = num_init_steps
        self.apply_collected_fn = apply_collected_fn

    def add_collectors(self, device):
        raise NotImplementedError

    def remove_collectors(self):
        raise NotImplementedError

    def run(self, model, data_loader, *args, **kwargs):
        class TQDMStream:
            @classmethod
            def write(cls, msg):
                tqdm.write(msg, end='')

        stream_handler = logging.StreamHandler(TQDMStream)
        logger.addHandler(stream_handler)
        device = next(model.parameters()).device
        self.add_collectors(device)
        # pylint: disable=unidiomatic-typecheck
        if type(data_loader) == torch.utils.data.DataLoader:
            wrapped_data_loader = InitializeDataLoader(data_loader=data_loader,
                                                       kwargs=kwargs,
                                                       device=device)
        else:
            wrapped_data_loader = data_loader
        with torch.no_grad():
            bar_format = '{l_bar}{bar} |{n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            bar_desc = 'Algorithm initialization'
            for i, (input_, dataloader_kwargs) in tqdm(enumerate(wrapped_data_loader), total=self.num_init_steps,
                                                       desc=bar_desc, bar_format=bar_format):
                if self.num_init_steps is not None and i >= self.num_init_steps:
                    break
                model(input_, **dataloader_kwargs)
            logger.removeHandler(stream_handler)
            self.apply_collected_fn(self, *args)
        self.remove_collectors()


class MinMaxInitializer(DataLoaderInitializer):
    MIN_ATTR_NAME = 'min_value'
    MAX_ATTR_NAME = 'max_value'

    def __init__(self, modules_to_init, apply_collected_fn, num_init_steps):
        super().__init__(apply_collected_fn, num_init_steps)
        self.collectors = []
        self.modules_to_init = modules_to_init

    def add_collectors(self, device):
        for module in self.modules_to_init.values():
            setattr(module, self.MIN_ATTR_NAME, torch.tensor(np.inf).to(device))
            setattr(module, self.MAX_ATTR_NAME, torch.tensor(-np.inf).to(device))
            self.collectors.append(module.register_forward_hook(self.forward_pre_hook))

    def remove_collectors(self):
        for handle in self.collectors:
            handle.remove()
        for module in self.modules_to_init.values():
            delattr(module, self.MIN_ATTR_NAME)
            delattr(module, self.MAX_ATTR_NAME)

    @classmethod
    def get_max_value(cls, module):
        return getattr(module, cls.MAX_ATTR_NAME)

    @classmethod
    def get_min_value(cls, module):
        return getattr(module, cls.MIN_ATTR_NAME)

    @staticmethod
    def forward_pre_hook(module, inputs, output):
        for input_ in inputs:
            max_value = torch.max(input_.abs().max(), MinMaxInitializer.get_max_value(module))
            min_value = torch.min(input_.min(), MinMaxInitializer.get_min_value(module))
        MinMaxInitializer.get_max_value(module).set_(max_value)
        MinMaxInitializer.get_min_value(module).set_(min_value)
