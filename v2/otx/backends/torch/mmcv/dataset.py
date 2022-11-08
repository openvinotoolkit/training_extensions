from abc import abstractmethod

from otx.api.dataset import Dataset
from otx.backends.torch.dataset import TorchDatasetAdapter
from otx.utils.logger import get_logger

logger = get_logger()


class MMDatasetAdapter(TorchDatasetAdapter):
    @abstractmethod
    def convert(self, src_dataset):
        raise NotImplementedError()

    def build(self, dataset: Dataset, subset: str):
        """ build dataset for backend by converting from given datumaro dataset
        """
        logger.info(f"dataset = {dataset}, subset = {subset}")
        logger.info(f"subset config = {self.config[subset]}")
        sub_dataset = dataset.get_subset(subset)

        # convert to backend consumable dataset from sub_dataset
        return self.convert(sub_dataset)
