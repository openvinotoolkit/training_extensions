# Copyright (c) OpenMMLab. All rights reserved.
from .bbox_head import BBoxHead
from .convfc_bbox_head import ConvFCBBoxHead, Shared2FCBBoxHead

__all__ = [
    'BBoxHead', 'ConvFCBBoxHead', 'Shared2FCBBoxHead',
]
