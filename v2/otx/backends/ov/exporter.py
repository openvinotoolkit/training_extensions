from otx.backends.ov.job import OpenVINOJob
from otx.utils.config import Config
from otx.utils.logger import get_logger

logger = get_logger()


class OpenVINOExporter(OpenVINOJob):
    def configure(self, cfg: Config, model_cfg=None, data_cfg=None, **kwargs):
        logger.info(f"configure({cfg})")
        training = kwargs.get("training", True)
        return cfg

    def run(self, model, type, task_config, **kwargs):
        logger.info(f"{__name__} run(model = {model}, type = {type}, config = {task_config}, others = {kwargs})")
        task_config = self.configure(task_config)
        metric = kwargs.get("metric", "top-1")
        return dict(export_path="path/to/export/results")
