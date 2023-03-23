# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.algorithms.common.utils.logger import get_logger
from otx.mpa.cls.stage import ClsStage

logger = get_logger()


class SemiSLClsStage(ClsStage):
    """Patch config to support semi supervised learning for object Cls"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def configure_data(self, cfg, data_cfg, training, **kwargs):
        """Patch cfg.data."""
        super().configure_data(cfg, data_cfg, training, **kwargs)
        # Set unlabeled data hook
        if training:
            if cfg.data.get("unlabeled", False) and cfg.data.unlabeled.get("otx_dataset", False):
                self.configure_unlabeled_dataloader(cfg, self.distributed)
