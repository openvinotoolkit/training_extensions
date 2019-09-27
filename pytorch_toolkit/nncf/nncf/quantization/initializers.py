import logging
from collections import OrderedDict

from nncf.utils import get_all_modules_by_type
from ..initializers import MinMaxInitializer
from ..registry import Registry

logger = logging.getLogger(__name__)

QUANTIZATION_INITIALIZERS = Registry('quantization_initializers')
MIN_MAX_INITIALIZERS = Registry('min_max_quantize_initializers')


@QUANTIZATION_INITIALIZERS.register('min_max')
class QuantizeMinMaxInitializer:
    def __init__(self, model, num_init_steps):
        self.model = model

        def apply_collected_fn(initializer, modules_to_init_, distributed_):
            for name, module in modules_to_init_.items():
                if hasattr(module, 'initialized'):
                    if module.initialized:
                        continue
                max_value = initializer.get_max_value(module)
                min_value = initializer.get_min_value(module)
                module_initializer = MIN_MAX_INITIALIZERS.get(type(module).__name__)
                module_initializer(module, name, min_value, max_value, distributed_)

        self.modules_to_init = OrderedDict()
        for module_type, _ in MIN_MAX_INITIALIZERS.registry_dict.items():
            self.modules_to_init.update(get_all_modules_by_type(self.model, module_type))
        # NOTE: Order of modules must be the same to correctly broadcast parameters (e.g. input_low and input_range)
        self.modules_to_init = OrderedDict(sorted(self.modules_to_init.items()))
        self.initializer = MinMaxInitializer(self.modules_to_init, apply_collected_fn, num_init_steps)

    def run(self, data_loader, is_distributed):
        if self.modules_to_init:
            for module in self.modules_to_init.values():
                module.init_stage = True
            self.initializer.run(self.model, data_loader, self.modules_to_init, is_distributed)
            for module in self.modules_to_init.values():
                module.init_stage = False
