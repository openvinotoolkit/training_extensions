from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.yolox_head import YOLOXHead


@HEADS.register_module()
class CustomYOLOXHead(YOLOXHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
