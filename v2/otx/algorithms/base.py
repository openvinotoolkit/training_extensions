from otx.api.task import Task
from otx.utils import import_and_get_class_from_path
from otx.utils.logger import get_logger

logger = get_logger()


class BaseTask(Task):
    def __init__(self, config):
        super().__init__(config)

    def _init_model_adapter(self, model_cfg):
        clz = import_and_get_class_from_path(model_cfg.adapter)
        self.model_adapter = clz(model_cfg.default)

    def _init_dataset_adapter(self, data_cfg):
        clz = import_and_get_class_from_path(data_cfg.adapter)
        self.dataset_adapter = clz(data_cfg.default)
