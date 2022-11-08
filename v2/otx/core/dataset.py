from abc import abstractmethod

from otx.api.dataset import Dataset
from otx.utils.config import Config
from otx.utils.logger import get_logger

logger = get_logger()


class IDatasetAdapter:
    def __init__(self, config_yaml: str):
        logger.info(f"config_yaml = {config_yaml}")
        self.config = Config.fromfile(config_yaml)

    @abstractmethod
    def build(self, dataset: Dataset, subset: str):
        raise NotImplementedError()
