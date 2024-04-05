# Copyright (c) OpenMMLab. All rights reserved.
from .bbox_nms import fast_nms, multiclass_nms


# yapf: enable

__all__ = [
    'fast_nms', 'multiclass_nms',
]
