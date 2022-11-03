from otx.utils.config import Config
from otx.utils.logger import get_logger
from otx.backends.torch.model import TorchModel

logger = get_logger()

class ModelAdapter(TorchModel):
    def build(self):
        """ create a model consumable by anomalib
        """
        model_name = self._config.model.name
        logger.info(f"build model for {model_name}")
        return None