import os


from otx.backends.torch.mmcv.job import MMJob
from otx.utils.config import Config
from otx.utils.logger import get_logger

logger = get_logger()


class MMClsJob(MMJob):
    def configure(self, cfg: Config, model_cfg=None, data_cfg=None, **kwargs):
        logger.info(f"configure({cfg})")
        training = kwargs.get("training", True)
        return cfg
