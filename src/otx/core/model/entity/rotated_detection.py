# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for rotated detection model entity used in OTX."""


from otx.core.model.entity.instance_segmentation import (
    MMDetInstanceSegCompatibleModel,
    OTXInstanceSegModel,
    OVInstanceSegmentationModel,
)


class OTXRotatedDetModel(OTXInstanceSegModel):
    """Base class for the rotated detection models used in OTX."""


class MMDetRotatedDetModel(OTXRotatedDetModel, MMDetInstanceSegCompatibleModel):
    """Rotated Detection model compaible for MMDet."""


class OVRotatedDetectionModel(OVInstanceSegmentationModel):
    """Rotated Detection model compatible for OpenVINO IR Inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the OTX detection model compatible for OTX testing pipeline.
    """
