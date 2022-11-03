from otx.api.dataset import Dataset
from otx.algorithms.anomaly.tasks import AnomalyTask


class AnomalyDetTask(AnomalyTask):
    def train(self, dataset: Dataset):
        datamodule = self.dataset_adapter.convert(dataset.build())
        model = self.model_adapter.build()

        return self.jobs["train"].run(model, datamodule, config=self.config.train)