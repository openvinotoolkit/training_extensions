import random

import numpy as np
import torch

from otx.core.job import IJob
from otx.utils.logger import get_logger

logger = get_logger()

class TorchJob(IJob):
    def __init__(self, spec, **kwargs):
        super().__init__(spec)

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
        return torch.cuda.is_available()

    @staticmethod
    def get_current_device():
        return torch.cuda.current_device()
