"""Custom YOLOX head for OTX template."""
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.yolox_head import YOLOXHead


@HEADS.register_module()
class CustomYOLOXHead(YOLOXHead):
    """CustomYOLOXHead class for OTX."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
