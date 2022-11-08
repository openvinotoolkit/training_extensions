from otx.api.dataset import Dataset
from otx.backends.torch.mmcv.dataset import MMDatasetAdapter
from otx.utils.config import Config
from otx.utils.logger import get_logger

logger = get_logger()


class MMClsDatasetAdapter(MMDatasetAdapter):
    def convert(self, src_dataset):
        logger.info(f"src dataset = {src_dataset}")
        dataset = None
        return dataset
