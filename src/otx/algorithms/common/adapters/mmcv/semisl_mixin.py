"""SemiSL configuration mixin."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import importlib

from mmcv.utils import Config, ConfigDict

from otx.algorithms.common.adapters.mmcv.utils import (
    build_dataloader,
    build_dataset,
)


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

        model_task = {"classification": "mmcls", "detection": "mmdet", "segmentation": "mmseg"}
        if "unlabeled" in cfg.data:
            task_lib_module = importlib.import_module(f"{model_task[cfg.model_task]}.datasets")
            dataset_builder = getattr(task_lib_module, "build_dataset")
            dataloader_builder = getattr(task_lib_module, "build_dataloader")

            dataset = build_dataset(cfg, "unlabeled", dataset_builder, consume=True)
            unlabeled_dataloader = build_dataloader(
                dataset,
                cfg,
                "unlabeled",
                dataloader_builder,
                distributed=cfg.distributed,
                consume=True,
            )

            custom_hooks = cfg.get("custom_hooks", [])
            updated = False
            for custom_hook in custom_hooks:
                if custom_hook["type"] == "ComposedDataLoadersHook":
                    custom_hook["data_loaders"] = [*custom_hook["data_loaders"], unlabeled_dataloader]
                    updated = True
            if not updated:
                custom_hooks.append(
                    ConfigDict(
                        type="ComposedDataLoadersHook",
                        data_loaders=unlabeled_dataloader,
                    )
                )
            cfg.custom_hooks = custom_hooks
