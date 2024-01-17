# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""DeitTiny model implementation."""

from otx.algo.utils.mmconfig import read_mmconfig
from otx.core.model.entity.classification import (
    MMPretrainHlabelClsModel,
    MMPretrainMulticlassClsModel,
    MMPretrainMultilabelClsModel,
)


class DeitTinyForHLabelCls(MMPretrainHlabelClsModel):
    """DeitTiny Model for hierarchical label classification task."""

    def __init__(self, num_classes: int, num_multiclass_heads: int, num_multilabel_classes: int) -> None:
        config = read_mmconfig(model_name="deit_tiny", subdir_name="hlabel_classification")
        config.head.num_multiclass_heads = num_multiclass_heads
        config.head.num_multilabel_classes = num_multilabel_classes
        super().__init__(num_classes=num_classes, config=config)


class DeitTinyForMulticlassCls(MMPretrainMulticlassClsModel):
    """DeitTiny Model for multi-label classification task."""

    def __init__(self, num_classes: int) -> None:
        config = read_mmconfig("deit_tiny", subdir_name="multiclass_classification")
        super().__init__(num_classes=num_classes, config=config)


class DeitTinyForMultilabelCls(MMPretrainMultilabelClsModel):
    """DeitTiny Model for multi-class classification task."""

    def __init__(self, num_classes: int) -> None:
        config = read_mmconfig("deit_tiny", subdir_name="multilabel_classification")
        super().__init__(num_classes=num_classes, config=config)
