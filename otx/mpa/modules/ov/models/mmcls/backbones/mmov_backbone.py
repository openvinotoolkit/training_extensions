# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import List, Dict

from mmcls.models.builder import BACKBONES

from ...mmov_model import MMOVModel
from ....graph.parsers.cls.cls_base_parser import cls_base_parser


@BACKBONES.register_module()
class MMOVBackbone(MMOVModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def parser(graph, **kwargs) -> Dict[str, List[str]]:
        output = cls_base_parser(graph, "backbone")
        if output is None:
            raise ValueError(
                "Parser can not determine input and output of model. "
                "Please provide them explicitly"
            )
        return output
