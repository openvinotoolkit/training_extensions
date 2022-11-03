from otx.api.dataset import Dataset
from otx.backends.torch.dataset import DatasetAdapter
from otx.utils.logger import get_logger

logger = get_logger()

class MMClsDataAdapter(DatasetAdapter):
    def convert(self, dataset: Dataset, ):
        self.source = dataset
        logger.info(f"[{__file__}] convert({dataset}) to mmcls dataset")
