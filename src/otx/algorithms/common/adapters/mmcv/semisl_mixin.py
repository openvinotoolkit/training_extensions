"""SemiSL configuration mixin."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.utils import Config, ConfigDict


class SemiSLConfigurerMixin:
    """Patch config to support semi supervised learning."""

    def configure_data_pipeline(self, cfg, input_size, model_ckpt_path, **kwargs):
        """Patch cfg.data."""
        super().configure_data_pipeline(cfg, input_size, model_ckpt_path, **kwargs)
        # Set unlabeled data hook
        if self.training:
            if cfg.data.get("unlabeled", False) and cfg.data.unlabeled.get("otx_dataset", False):
                if len(cfg.data.unlabeled.get("pipeline", [])) == 0:
                    cfg.data.unlabeled.pipeline = cfg.data.train.pipeline.copy()
                self.configure_unlabeled_dataloader(cfg)

    @staticmethod
    def configure_unlabeled_dataloader(cfg: Config):
        """Patch for unlabled dataloader."""
        if "unlabeled" in cfg.data:
            custom_hooks = cfg.get("custom_hooks", [])
            custom_hooks.append(
                ConfigDict(
                    type="ComposedDataLoadersHook",
                    cfg=cfg,
                )
            )
            cfg.custom_hooks = custom_hooks
