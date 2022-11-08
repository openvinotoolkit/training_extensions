from otx.backends.torch.job import TorchJob
from otx.utils.config import Config
from otx.utils.logger import get_logger

logger = get_logger()


class AnomalibJob(TorchJob):
    def configure(self, config: Config, **kwargs):
        """ job specific configuration update
        """
        logger.info(f"config = {config}, kwargs = {kwargs}")
