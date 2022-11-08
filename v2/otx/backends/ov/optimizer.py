from otx.backends.ov.job import OpenVINOJob
from otx.utils.logger import get_logger

logger = get_logger()


class OpenVINOOptimizer(OpenVINOJob):
    def run(self, model, opt_type, task_config, **kwargs):
        logger.info(f"model = {model}, opt_type = {opt_type}, task_config = {task_config}, kwargs = {kwargs}")
        return dict(optimized="path/to/optimized/results")
