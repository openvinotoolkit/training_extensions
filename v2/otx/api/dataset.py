import os
from abc import abstractmethod

from otx.utils.config import Config
from otx.utils.logger import get_logger

import datumaro

logger = get_logger()


class Dataset(datumaro.Dataset):
    def __init__(self, dataset: datumaro.Dataset, **kwargs):
        super().__init__(source=dataset)
        logger.info(f"dataset = {dataset}, kwargs = {kwargs}")
        self.spec = kwargs.get("spec", "unknown")

    @staticmethod
    def create(path: str, format: str, **kwargs):
        logger.info(f"path = {path}, format = {format}, kwargs = {kwargs}")
        dm = datumaro.Dataset.import_from(path, format, **kwargs)
        return Dataset(dm)
