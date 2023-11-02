"""OTX encoder decoder for semantic segmentation."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmseg.models import SEGMENTORS
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from otx.v2.api.utils.logger import get_logger

logger = get_logger()


@SEGMENTORS.register_module()
class OTXEncoderDecoder(EncoderDecoder):
    """OTX encoder decoder."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
