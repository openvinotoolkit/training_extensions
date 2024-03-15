"""Object-detection-related modules for otx.core.ov.ops."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import List, Optional

from otx.core.ov.ops.builder import OPS
from otx.core.ov.ops.op import Attribute, Operation

# pylint: disable=too-many-instance-attributes


@dataclass
class ProposalV4Attribute(Attribute):
    """ProposalV4Attribute class."""

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
        """ProposalV4Attribute's post-init function."""
        super().__post_init__()
        valid_framework = ["", "tensorflow"]
        if self.framework not in valid_framework:
            raise ValueError(f"Invalid framework {self.framework}. " f"It must be one of {valid_framework}.")


@OPS.register()
class ProposalV4(Operation[ProposalV4Attribute]):
    """ProposalV4 class."""

    TYPE = "Proposal"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = ProposalV4Attribute

    def forward(self, class_probs, bbox_deltas, image_shape):
        """ProposalV4's forward function."""
        raise NotImplementedError


@dataclass
class ROIPoolingV0Attribute(Attribute):
    """ROIPoolingV0Attribute class."""

    pooled_h: int
    pooled_w: int
    spatial_scale: float
    method: str = field(default="max")
    output_size: List[int] = field(default_factory=lambda: [])

    def __post_init__(self):
        """ROIPoolingV0Attribute's post-init function."""
        super().__post_init__()
        valid_method = ["max", "bilinear"]
        if self.method not in valid_method:
            raise ValueError(f"Invalid method {self.method}. " f"It must be one of {valid_method}.")


@OPS.register()
class ROIPoolingV0(Operation[ROIPoolingV0Attribute]):
    """ROIPoolingV0 class."""

    TYPE = "ROIPooling"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = ROIPoolingV0Attribute

    def forward(self, inputs, boxes):
        """ROIPoolingV0's forward function."""
        raise NotImplementedError


@dataclass
class DetectionOutputV0Attribute(Attribute):
    """DetectionOutputV0Attribute class."""

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
        """DetectionOutputV0Attribute's post-init function."""
        super().__post_init__()
        valid_code_type = [
            "caffe.PriorBoxParameter.CORNER",
            "caffe.PriorBoxParameter.CENTER_SIZE",
        ]
        if self.code_type not in valid_code_type:
            raise ValueError(f"Invalid code_type {self.code_type}. " f"It must be one of {valid_code_type}.")


@OPS.register()
class DetectionOutputV0(Operation[DetectionOutputV0Attribute]):
    """DetectionOutputV0 class."""

    TYPE = "DetectionOutput"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = DetectionOutputV0Attribute

    def forward(self, loc_data, conf_data, prior_data, arm_conf_data=None, arm_loc_data=None):
        """DetectionOutputV0's forward."""
        raise NotImplementedError


@dataclass
class RegionYoloV0Attribute(Attribute):
    """RegionYoloV0Attribute class."""

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
    """RegionYoloV0 class."""

    TYPE = "RegionYolo"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = RegionYoloV0Attribute

    def forward(self, inputs):
        """RegionYoloV0's forward function."""
        raise NotImplementedError


@dataclass
class PriorBoxV0Attribute(Attribute):
    """PriorBoxV0Attribute class."""

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
    """PriorBoxV0 class."""

    TYPE = "PriorBox"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = PriorBoxV0Attribute

    def forward(self, output_size, image_size):
        """PriorBoxV0's forward function."""
        raise NotImplementedError


@dataclass
class PriorBoxClusteredV0Attribute(Attribute):
    """PriorBoxClusteredV0Attribute class."""

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
    """PriorBoxClusteredV0 class."""

    TYPE = "PriorBoxClustered"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = PriorBoxClusteredV0Attribute

    def forward(self, output_size, image_size):
        """PriorBoxClusteredV0's forward function."""
        raise NotImplementedError
