from otx.backends.torch.mmcv.job import MMJob
from otx.utils.config import Config
from otx.utils.logger import get_logger

logger = get_logger()


class MMEvaluator(MMJob):
    def __init__(self, spec, **kwargs):
        super().__init__(spec, **kwargs)
        logger.info(f"{__name__} __init__({spec}, {kwargs})")

    def configure(self, cfg: Config, model_cfg=None, data_cfg=None, **kwargs):
        logger.info(f"configure({cfg})")
        training = kwargs.get("training", True)
        return cfg

    def run(self, model, data, task_config, **kwargs):
        logger.info(f"{__name__} run(model = {model}, datasets = {data}, config = {task_config}, others = {kwargs})")
        task_config = self.configure(task_config)
        metric = kwargs.get("metric", "top-1")
        return dict(result=None)
