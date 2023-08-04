"""Collection of utils for task implementation in Segmentation Task."""
# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import logging
from typing import List, Optional

from mmcv import Config, ConfigDict

from otx.algorithms.common.adapters.mmcv.utils import (
    get_configs_by_pairs,
    get_dataset_configs,
    get_meta_keys,
    patch_color_conversion,
    remove_from_config,
)
from otx.api.entities.label import Domain

logger = logging.getLogger(__name__)


def patch_datasets(
    config: Config,
    domain: Domain = Domain.SEGMENTATION,
    subsets: Optional[List[str]] = None,
    **kwargs,
):
    """Update dataset configs."""
    assert "data" in config
    assert "type" in kwargs

    if subsets is None:
        subsets = ["train", "val", "test", "unlabeled"]

    def update_pipeline(cfg, subset):
        if subset == "train":
            for collect_cfg in get_configs_by_pairs(cfg, dict(type="Collect")):
                get_meta_keys(collect_cfg)
        for cfg_ in get_configs_by_pairs(cfg, dict(type="LoadImageFromFile")):
            cfg_.type = "LoadImageFromOTXDataset"
            if subset != "test":
                cfg_.enable_memcache = True
        for cfg_ in get_configs_by_pairs(cfg, dict(type="LoadAnnotations")):
            cfg_.type = "LoadAnnotationFromOTXDataset"

    for subset in subsets:
        if subset not in config.data:
            continue
        config.data[f"{subset}_dataloader"] = config.data.get(f"{subset}_dataloader", ConfigDict())

        cfgs = get_dataset_configs(config, subset)
        for cfg in cfgs:
            cfg.domain = domain
            cfg.otx_dataset = None
            cfg.labels = None
            cfg.update(kwargs)

            remove_from_config(cfg, "ann_dir")
            remove_from_config(cfg, "img_dir")
            remove_from_config(cfg, "data_root")
            remove_from_config(cfg, "split")
            remove_from_config(cfg, "classes")

            update_pipeline(cfg, subset)

    patch_color_conversion(config)


def patch_evaluation(config: Config):
    """Update evaluation configs."""
    cfg = config.get("evaluation", None)
    if cfg:
        cfg.pop("classwise", None)
        cfg.metric = "mDice"
        cfg.save_best = "mDice"
        cfg.rule = "greater"
        # EarlyStoppingHook
        config.early_stop_metric = "mDice"
