import logging

from torch import distributed

from ..initializers import MinMaxInitializer
from ..registry import Registry

logger = logging.getLogger(__name__)

QUANTIZATION_INITIALIZERS = Registry('quantization_initializers')


@QUANTIZATION_INITIALIZERS.register('min_max')
class QuantizeMinMaxInitializer:
    def __init__(self, modules_to_init, num_init_steps):
        def apply_collected_fn(initializer, modules_to_init_, distributed_):
            for name, module in modules_to_init_.items():
                max_value = initializer.get_max_value(module)
                min_value = initializer.get_min_value(module)
                if abs(max_value) > 0.1:
                    module.scale.data.fill_(max_value.item())
                sign = min_value.item() < 0
                if sign != module.signed:
                    logger.warning("signed set incorrectly")
                module.signed = int(sign)
                if distributed_:
                    distributed.broadcast(module.scale, 0)
                    distributed.broadcast(module.signed_tensor, 0)
                logger.debug("Statistics: min={:.2f} max={:.2f}".format(min_value.item(), max_value.item()))
                logger.info(
                    "Set sign: {} and scale: {:04.2f} for {}".format(module.signed, module.scale.item(), name))

        self.initializer = MinMaxInitializer(modules_to_init, apply_collected_fn, num_init_steps)
        self.modules_to_init = modules_to_init

    def run(self, model, data_loader, is_distributed):
        self.initializer.run(model, data_loader, self.modules_to_init, is_distributed)
