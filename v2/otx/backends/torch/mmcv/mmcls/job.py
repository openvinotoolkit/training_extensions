from mmcls.version import __version__

from otx.backends.torch.mmcv.job import MMJob
from otx.utils.config import Config
from otx.utils.logger import get_logger

logger = get_logger()


class MMClsJob(MMJob):
    def configure(self, task_config: Config, **kwargs):
        logger.info(f"task_config = {task_config}")
        return task_config

    @staticmethod
    def get_mmcls_version():
        logger.info(f"mmcls version = {__version__}")
        return __version__
