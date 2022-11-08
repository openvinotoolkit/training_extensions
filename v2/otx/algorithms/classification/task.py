from otx.algorithms.base import BaseTask
from otx.api.dataset import Dataset
from otx.core.model import ModelStatus
from otx.utils.logger import get_logger

logger = get_logger()


class ClsTask(BaseTask):
    def eval(self, dataset: Dataset, metric: str, **kwargs):
        logger.info(f"dataset = {dataset}, metric = {metric}, kwargs = {kwargs}")
        spec = kwargs.get("spec", "eval")
        logger.info("=== prepare model ===")
        model = self.model_adapter.build() if self.model is None else self.model
        infer_results = self.infer(dataset, **kwargs)
        return self._run_job(spec, model, metric=metric, infer_results=infer_results, **kwargs)

    def infer(self, dataset: Dataset, **kwargs):
        logger.info(f"dataset = {dataset}, kwargs = {kwargs}")
        spec = kwargs.get("spec", "infer")
        model = self.model_adapter.build() if self.model is None else self.model
        dataset = self.dataset_adapter.build(dataset.get_subset("test"), "test")
        return self._run_job(spec, model, dataset=dataset, **kwargs)

    def optimize(self, opt_type, **kwargs):
        logger.info(f"opt_type = {opt_type}, kwargs = {kwargs}")
        spec = kwargs.get("spec", "optimize")
        model = self.model_adapter.build() if self.model is None else self.model
        ret = self._run_job(spec, model, opt_type, **kwargs)
        if ret is not None:
            self.model_status = ModelStatus.OPTIMIZED
        return ret


class ClsIncrClassification(ClsTask):
    def train(self, dataset: Dataset, **kwargs):
        logger.info(f"dataset = {dataset}, kwargs = {kwargs}")
        spec = kwargs.get("spec", "train")

        logger.info("=== prepare model ===")
        self.model = self.model_adapter.build()
        logger.info("=== prepare dataset ===")
        datasets = dict(
            train=self.dataset_adapter.build(dataset, "train"),
            val=self.dataset_adapter.build(dataset, "val"),
        )
        results = self._run_job(spec, self.model, datasets, **kwargs)
        ckpt = results.get("final_ckpt")
        if ckpt is not None:
            self.model_adapter.ckpt = ckpt
            self.model_status = ModelStatus.TRAINED
        return results
