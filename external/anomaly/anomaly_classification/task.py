from ote_anomalib import BaseAnomalyTask
from omegaconf import ListConfig, DictConfig
from typing import Union


class AnomalyClassificationTask(BaseAnomalyTask):

    def get_config(self) -> Union[DictConfig, ListConfig]:
        config = super().get_config()
        config.dataset.task = "classification"
        return config
