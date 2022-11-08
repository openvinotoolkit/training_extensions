from abc import abstractmethod

from otx.utils.config import Config
from otx.utils.logger import get_logger

logger = get_logger()


class IJob:
    """"""
    def __init__(self, spec, config: Config):
        """"""
        self.spec = spec
        self.config = config
        logger.info(f"spec = {spec}, config = {config}")

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def configure(self, task_config: Config, **kwargs):
        """ update task configuration w.r.t the job implementation
        """
        raise NotImplementedError()
