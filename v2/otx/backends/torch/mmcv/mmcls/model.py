from mmcls.models.builder import build_classifier

from otx.backends.torch.mmcv.model import MMModelAdapter
from otx.utils.logger import get_logger

logger = get_logger()


class MMClsModelAdapter(MMModelAdapter):
    def __init__(self, config_yaml: str):
        super().__init__(config_yaml)
        logger.info(f"config_yaml = {config_yaml}")

    def build(self, **kwargs):
        logger.info(f"kwargs = {kwargs}")
        logger.info(f"building a model with config = {self._config}")
        model = build_classifier(self._config._cfg_dict)
        if self._ckpt is not None:
            # load weights from ckpt
            model.load_state_dict(self.load_weights_from_ckpt(self._ckpt))
        return model
