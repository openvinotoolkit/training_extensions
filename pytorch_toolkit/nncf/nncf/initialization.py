import logging
from typing import Dict, Tuple
from collections import OrderedDict

import torch
from tqdm import tqdm

from nncf.registry import Registry
from nncf.quantization.initializers import InitializerFactory

logger = logging.getLogger(__name__)

INITIALIZABLE_MODULES = Registry('initializable_modules')

def wrap_data_loader(data_loader, kwargs=None, device=None):
    # pylint: disable=unidiomatic-typecheck
    if type(data_loader) == torch.utils.data.DataLoader:
        wrapped_data_loader = InitializingDataLoader(data_loader=data_loader,
                                                     kwargs=kwargs,
                                                     device=device)
    else:
        wrapped_data_loader = data_loader
    return wrapped_data_loader


class InitializingDataLoader:
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


class DataLoaderInitializeRunner:
    def __init__(self, model, modules_to_init: Dict[str, Tuple[torch.nn.Module, str]]):
        super().__init__()
        self.model = model
        self.modules_to_init = modules_to_init

    def run(self, data_loader, num_init_steps, is_distributed, *args, **kwargs):
        class TQDMStream:
            @classmethod
            def write(cls, msg):
                tqdm.write(msg, end='')

        stream_handler = logging.StreamHandler(TQDMStream)
        logger.addHandler(stream_handler)
        device = next(self.model.parameters()).device
        wrapped_data_loader = wrap_data_loader(data_loader, kwargs, device)

        initializers = OrderedDict()
        hook_handles = []
        for name, data in self.modules_to_init.items():
            module, init_type = data
            initializers[name] = InitializerFactory.create(init_type)(module,
                                                                      is_distributed,
                                                                      log_module_name=name)
            hook_handles.append(module.register_forward_hook(initializers[name].forward_hook))
            module.init_stage = True
        with torch.no_grad():
            bar_format = '{l_bar}{bar} |{n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            bar_desc = 'Algorithm initialization'
            for i, (input_, dataloader_kwargs) in tqdm(enumerate(wrapped_data_loader), total=num_init_steps,
                                                       desc=bar_desc, bar_format=bar_format):
                if num_init_steps is not None and i >= num_init_steps:
                    break
                self.model(input_, **dataloader_kwargs)
            logger.removeHandler(stream_handler)
            for handle in hook_handles:
                handle.remove()
            for initializer in initializers.values():
                initializer.apply_init()

        for module, _ in self.modules_to_init.values():
            module.init_stage = False
