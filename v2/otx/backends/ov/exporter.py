from otx.backends.ov.job import OpenVINOJob
from otx.utils.logger import get_logger

logger = get_logger()


class OpenVINOExporter(OpenVINOJob):
    def run(self, model, exp_type, task_config, **kwargs):
        logger.info(f"model = {model}, exp_type = {exp_type}, task_config = {task_config}, kwargs = {kwargs}")
        return dict(export_path="path/to/export/results")
