from abc import abstractmethod
from enum import IntEnum

from mmcv.utils import Registry, build_from_cfg

from otx.utils.config import Config


class IJob():
    """"""
    def __init__(self, spec):
        """"""
        self.spec = spec

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def configure(self, config: Config, **kwargs):
        """ job specific configuration update
        """
        raise NotImplementedError()