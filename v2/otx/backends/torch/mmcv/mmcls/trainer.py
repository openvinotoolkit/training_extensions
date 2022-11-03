import numbers
import os
import time

from mmcls.version import __version__

from otx.backends.torch.mmcv.mmcls.job import MMClsJob
from otx.utils.logger import get_logger

logger = get_logger()


class MMClsTrainer(MMClsJob):
    def __init__(self, spec, **kwargs):
        super().__init__(spec, **kwargs)
        logger.info(f"{__name__} __init__({spec}, {kwargs})")

    def run(self, model, data, task_config, **kwargs):
        logger.info(f"{__name__} run(model = {model}, datasets = {data}, config = {task_config}, others = {kwargs})")
        task_config = self.configure(task_config)
        logger.info(f"run with mmcls {__version__} backend")
        logger.info(f"is gpu available = {self.is_gpu_available()}")
        return dict(final_ckpt=output_ckpt_path)
