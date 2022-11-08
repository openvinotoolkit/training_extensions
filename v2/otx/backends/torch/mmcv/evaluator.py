from otx.backends.torch.mmcv.job import MMJob
from otx.utils.config import Config
from otx.utils.logger import get_logger

logger = get_logger()


class MMEvaluator(MMJob):
    def configure(self, task_config: Config, **kwargs):
        logger.info(f"task_config = {task_config}, kwargs = {kwargs}")
        return task_config

    def run(self, model, task_config=None, **kwargs):
        logger.info(f"model = {model}, task_config = {task_config}, kwargs = {kwargs}")
        if task_config is not None:
            task_config = self.configure(task_config)
        if self.spec == "eval":
            infer_results = kwargs.get("infer_results")
            metric = kwargs.get("metric")
            return self.eval(infer_results, metric)
        dataset = kwargs.get("dataset")
        return self.infer(dataset)

    def infer(self, dataset):
        logger.info(f"dataset = {dataset}")
        return dict(
            infer="result"
        )

    def eval(self, results, metric):
        logger.info(f"results = {results}, metric = {metric}")
        return dict(
            metric="good"
        )
