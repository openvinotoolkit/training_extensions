from otx.algorithms.base import BaseTask


class AnomalyTask(BaseTask):
    def train(self, dataset: Dataset, **kwargs):
        logger.info(f"dataset = {dataset}, kwargs = {kwargs}")
        spec = kwargs.get("spec", "train")
        datamodule = self.dataset_adapter.build(dataset, "train")
        model = self.model_adapter.build()

        results = self._run_job(spec, model, datasets, **kwargs)
        return results