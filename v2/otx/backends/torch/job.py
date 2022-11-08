import random

import numpy as np
import torch

from otx.core.job import IJob
from otx.utils.logger import get_logger

logger = get_logger()


class TorchJob(IJob):
    @staticmethod
    def _set_random_seed(seed, deterministic=False):
        """Set random seed.

        Args:
            seed (int): Seed to be used.
            deterministic (bool): Whether to set the deterministic option for
                CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
                to True and `torch.backends.cudnn.benchmark` to False.
                Default: False.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        logger.info(f'Training seed was set to {seed} w/ deterministic={deterministic}.')
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @staticmethod
    def is_gpu_available():
        logger.info(f"cuda available = {torch.cuda.is_available()}")
        return torch.cuda.is_available()

    @staticmethod
    def get_current_device():
        logger.info(f"current device = {torch.cuda.current_device()}")
        return torch.cuda.current_device()

    @staticmethod
    def get_torch_version():
        logger.info(f"torch version = {torch.version.__version__}")
        return torch.version.__version__
