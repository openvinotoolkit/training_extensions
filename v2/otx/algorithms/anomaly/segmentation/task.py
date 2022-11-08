from otx.api.dataset import Dataset
from otx.algorithms.anomaly.tasks import AnomalyTask


class AnomalySegTask(AnomalyTask):
    def eval(self, dataset: Dataset, metric: str, **kwargs):
        logger.info(f"dataset = {dataset}, metric = {metric}, kwargs = {kwargs}")
        spec = kwargs.get("spec", "eval")
        logger.info("=== prepare model ===")
        model = self.model_adapter.build() if self.model is None else self.model
        infer_results = self.infer(dataset, **kwargs)
        return self._run_job(spec, model, metric=metric, infer_results=infer_results, **kwargs)