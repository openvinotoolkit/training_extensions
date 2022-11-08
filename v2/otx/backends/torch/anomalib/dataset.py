from otx.api.dataset import Dataset
from otx.backends.torch.dataset import TorchDatasetAdapter
from otx.utils.logger import get_logger

logger = get_logger()

class AnomalibDatasetAdapter(TorchDatasetAdapter):
    def build(self, dataset: Dataset, subset: str):
        """ build dataset for backend by converting from given datumaro dataset
        """
        logger.info(f"dataset = {dataset}, subset = {subset}")
        logger.info(f"subset config = {self.config[subset]}")
        return None
