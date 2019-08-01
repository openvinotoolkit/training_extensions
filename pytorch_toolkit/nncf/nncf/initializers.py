import numpy as np
import torch


class DataLoaderInitializer:
    def __init__(self, apply_collected_fn, num_init_steps=None):
        super().__init__()
        self.num_init_steps = num_init_steps
        self.apply_collected_fn = apply_collected_fn

    def add_collectors(self, device):
        raise NotImplementedError

    def remove_collectors(self):
        raise NotImplementedError

    def run(self, model, data_loader, *args):
        device = next(model.parameters()).device
        self.add_collectors(device)
        with torch.no_grad():
            for i, (input_, _) in enumerate(data_loader):
                if self.num_init_steps is not None and i > self.num_init_steps:
                    break
                input_ = input_.to(device)
                model(input_)
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
