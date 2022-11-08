from otx.api.dataset import Dataset
from otx.backends.torch.data import DatasetAdapter
from otx.utils.logger import get_logger

logger = get_logger()

class AnomalibDataAdapter(DatasetAdapter):
    def build(self, dataset: Dataset, subset: str):
        """ build dataset for backend by converting from given datumaro dataset
        """
        logger.info(f"dataset = {dataset}, subset = {subset}")
        logger.info(f"subset config = {self.config[subset]}")
        return None
