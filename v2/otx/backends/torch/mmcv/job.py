from mmcv import build_from_cfg
from mmcv.version import __version__
from mmcv.runner.hooks import HOOKS

from otx.backends.torch.job import TorchJob
from otx.utils.logger import get_logger

logger = get_logger()


class MMJob(TorchJob):
    @staticmethod
    def build_checkpoint_hook(checkpoint_config):
        if checkpoint_config.get('type', False):
            hook = build_from_cfg(checkpoint_config, HOOKS)
        else:
            checkpoint_config.setdefault('type', 'CheckpointHook')
            hook = build_from_cfg(checkpoint_config, HOOKS)
        return hook

    @staticmethod
    def get_mmcv_version():
        logger.info(f"mmcv version = {__version__}")
        return __version__
