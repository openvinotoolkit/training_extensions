from mmcv import build_from_cfg
from mmcv.runner.hooks import HOOKS

from otx.backends.torch.job import TorchJob


class MMJob(TorchJob):
    def __init__(self, spec, **kwargs):
        super().__init__(spec, **kwargs)

    @staticmethod
    def register_checkpoint_hook(checkpoint_config):
        if checkpoint_config.get('type', False):
            hook = mmcv.build_from_cfg(checkpoint_config, HOOKS)
        else:
            checkpoint_config.setdefault('type', 'CheckpointHook')
            hook = mmcv.build_from_cfg(checkpoint_config, HOOKS)
        return hook
