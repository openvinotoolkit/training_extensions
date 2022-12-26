# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from dataclasses import dataclass, field
from typing import List, Optional

from .builder import OPS
from .op import Attribute, Operation


@dataclass
class ProposalV4Attribute(Attribute):
    base_size: int
    pre_nms_topn: int
    post_nms_topn: int
    nms_thresh: float
    feat_stride: int
    min_size: int
    ratio: List[float]
    scale: List[float]
    clip_before_nms: bool = field(default=True)
    clip_after_nms: bool = field(default=False)
    normalize: bool = field(default=False)
    box_size_scale: float = field(default=1.0)
    box_coordinate_scale: float = field(default=1.0)
    framework: str = field(default="")

    def __post_init__(self):
        super().__post_init__()
        valid_framework = ["", "tensorflow"]
        if self.framework not in valid_framework:
            raise ValueError(f"Invalid framework {self.framework}. " f"It must be one of {valid_framework}.")


@OPS.register()
class ProposalV4(Operation[ProposalV4Attribute]):
    TYPE = "Proposal"
    VERSION = 4
    ATTRIBUTE_FACTORY = ProposalV4Attribute

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #  from mmdet.core.anchor.anchor_generator import AnchorGenerator
        #  self._anchor_generator = AnchorGenerator(
        #      strides=[attrs["feat_stride"]],
        #      ratios=attrs["ratio"],
        #      scales=attrs["scale"],
        #      base_sizes=[attrs["base_size"]],
        #  )

        #  from torchvision.models.detection.anchor_utils import AnchorGenerator
        #  self._anchor_generator = AnchorGenerator(
        #      sizes=(self.attrs["base_size"],),
        #      aspect_ratios=

    def forward(self, class_probs, bbox_deltas, image_shape):
        raise NotImplementedError


@dataclass
class ROIPoolingV0Attribute(Attribute):
    pooled_h: int
    pooled_w: int
    spatial_scale: float
    method: str = field(default="max")
    output_size: List[int] = field(default_factory=lambda: [])

    def __post_init__(self):
        super().__post_init__()
        valid_method = ["max", "bilinear"]
        if self.method not in valid_method:
            raise ValueError(f"Invalid method {self.method}. " f"It must be one of {valid_method}.")


@OPS.register()
class ROIPoolingV0(Operation[ROIPoolingV0Attribute]):
    TYPE = "ROIPooling"
    VERSION = 0
    ATTRIBUTE_FACTORY = ROIPoolingV0Attribute

    def forward(self, input, boxes):
        raise NotImplementedError


@dataclass
class DetectionOutputV0Attribute(Attribute):
    keep_top_k: List[int]
    nms_threshold: float
    background_label_id: int = field(default=0)
    top_k: int = field(default=-1)
    variance_encoded_in_target: bool = field(default=False)
    code_type: str = field(default="caffe.PriorBoxParameter.CORNER")
    share_location: bool = field(default=True)
    confidence_threshold: float = field(default=0)
    clip_after_nms: bool = field(default=False)
    clip_before_nms: bool = field(default=False)
    decrease_label_id: bool = field(default=False)
    normalized: bool = field(default=False)
    input_height: int = field(default=1)
    input_width: int = field(default=1)
    objectness_score: float = field(default=0)

    def __post_init__(self):
        super().__post_init__()
        valid_code_type = [
            "caffe.PriorBoxParameter.CORNER",
            "caffe.PriorBoxParameter.CENTER_SIZE",
        ]
        if self.code_type not in valid_code_type:
            raise ValueError(f"Invalid code_type {self.code_type}. " f"It must be one of {valid_code_type}.")


@OPS.register()
class DetectionOutputV0(Operation[DetectionOutputV0Attribute]):
    TYPE = "DetectionOutput"
    VERSION = 0
    ATTRIBUTE_FACTORY = DetectionOutputV0Attribute

    def forward(self, loc_data, conf_data, prior_data, arm_conf_data=None, arm_loc_data=None):
        raise NotImplementedError


@dataclass
class RegionYoloV0Attribute(Attribute):
    axis: int
    coords: int
    classes: int
    end_axis: int
    num: int
    anchors: Optional[List[float]] = field(default=None)
    do_softmax: bool = field(default=True)
    mask: List[int] = field(default_factory=lambda: [])


@OPS.register()
class RegionYoloV0(Operation[RegionYoloV0Attribute]):
    TYPE = "RegionYolo"
    VERSION = 0
    ATTRIBUTE_FACTORY = RegionYoloV0Attribute

    def forward(self, input):
        raise NotImplementedError


@dataclass
class PriorBoxV0Attribute(Attribute):
    offset: float
    min_size: List[float] = field(default_factory=lambda: [])
    max_size: List[float] = field(default_factory=lambda: [])
    aspect_ratio: List[float] = field(default_factory=lambda: [])
    flip: bool = field(default=False)
    clip: bool = field(default=False)
    step: float = field(default=0)
    variance: List[float] = field(default_factory=lambda: [])
    scale_all_sizes: bool = field(default=True)
    fixed_ratio: List[float] = field(default_factory=lambda: [])
    fixed_size: List[float] = field(default_factory=lambda: [])
    density: List[float] = field(default_factory=lambda: [])


@OPS.register()
class PriorBoxV0(Operation[PriorBoxV0Attribute]):
    TYPE = "PriorBox"
    VERSION = 0
    ATTRIBUTE_FACTORY = PriorBoxV0Attribute

    def forward(self, output_size, image_size):
        raise NotImplementedError


@dataclass
class PriorBoxClusteredV0Attribute(Attribute):
    offset: float
    width: List[float] = field(default_factory=lambda: [1.0])
    height: List[float] = field(default_factory=lambda: [1.0])
    clip: bool = field(default=False)
    step: float = field(default=0.0)
    step_w: float = field(default=0.0)
    step_h: float = field(default=0.0)
    variance: List[float] = field(default_factory=lambda: [])


@OPS.register()
class PriorBoxClusteredV0(Operation[PriorBoxClusteredV0Attribute]):
    TYPE = "PriorBoxClustered"
    VERSION = 0
    ATTRIBUTE_FACTORY = PriorBoxClusteredV0Attribute

    def forward(self, output_size, image_size):
        raise NotImplementedError
