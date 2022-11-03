from abc import abstractmethod
from otx.api.dataset import Dataset
from otx.utils.config import Config
from otx.utils.logger import get_logger

logger = get_logger()

class DatasetAdapter(Dataset):
    def __init__(self, default_yaml, **kwargs):
        self.config = Config.fromfile(default_yaml)
        super().__init__(self.config)
