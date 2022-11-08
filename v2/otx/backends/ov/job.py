from otx.core.job import IJob
from otx.utils.logger import get_logger

logger = get_logger()


class OpenVINOJob(IJob):
    def get_version(self):
        logger.info("get_version()")
