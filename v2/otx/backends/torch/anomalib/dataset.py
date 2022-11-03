from otx.api.dataset import Dataset
from otx.backends.torch.data import DatasetAdapter
from otx.utils.logger import get_logger

logger = get_logger()

class AnomalibDataAdapter(DatasetAdapter):
    def convert(self, dataset: Dataset):
        self.source = dataset
        logger.info(f"[{__file__}] convert({dataset}) to anomalib dataset")
