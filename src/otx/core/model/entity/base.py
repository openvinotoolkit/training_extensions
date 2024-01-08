# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for detection model entity used in OTX."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic

from torch import nn

from otx.core.data.entity.base import (
    OTXBatchLossEntity,
    T_OTXBatchDataEntity,
    T_OTXBatchPredEntity,
)
from otx.core.data.entity.detection import DetBatchDataEntity, DetBatchPredEntity
from otx.core.data.entity.instance_segmentation import InstanceSegDataEntity, InstanceSegPredEntity
from otx.core.data.entity.tile import TileBatchDetDataEntity, TileBatchInstSegDataEntity
from otx.core.utils.tile_merge import merge


class OTXModel(nn.Module, Generic[T_OTXBatchDataEntity, T_OTXBatchPredEntity]):
    """Base class for the models used in OTX."""

    def __init__(self) -> None:
        super().__init__()
        self.model = self._create_model()

    @abstractmethod
    def _create_model(self) -> nn.Module:
        """Create a PyTorch model for this class."""

    def _customize_inputs(self, inputs: T_OTXBatchDataEntity) -> dict[str, Any]:
        """Customize OTX input batch data entity if needed for you model."""
        raise NotImplementedError

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: T_OTXBatchDataEntity,
    ) -> T_OTXBatchPredEntity | OTXBatchLossEntity:
        """Customize OTX output batch data entity if needed for you model."""
        raise NotImplementedError

    def unpack_tiles(
        self,
        inputs: TileBatchDetDataEntity | TileBatchInstSegDataEntity,
    ) -> DetBatchPredEntity | InstanceSegPredEntity:
        """Unpack tiles into batch data entity."""
        #TODO: Add support for InstanceSegPredEntity
        pred_entities = []
        for tiles, tile_infos, bboxes, labels in zip(
            inputs.batch_tiles,
            inputs.batch_tile_infos,
            inputs.bboxes,
            inputs.labels,
        ):
            tile_preds = []
            for tile, tile_info in zip(tiles, tile_infos):
                if isinstance(inputs, TileBatchDetDataEntity):
                    tile_input = DetBatchDataEntity(
                        image=tile,
                        img_info=tile_info,
                        bboxes=bboxes,
                        labels=labels,
                    )
                elif isinstance(inputs, TileBatchInstSegDataEntity):
                    tile_input = InstanceSegDataEntity(
                        image=tile,
                        img_info=tile_info,
                        bboxes=bboxes,
                        labels=labels,
                        polygons=inputs.polygons,
                    )
                tile_preds.append(self.forward(tile_input))
            pred_entities.append(merge(tile_preds))

        return DetBatchPredEntity(
            batch_size=inputs.batch_size,
            images=[entity.image for entity in pred_entities],
            imgs_info=[entity.img_info for entity in pred_entities],
            scores=[entity.score for entity in pred_entities],
            bboxes=[entity.bboxes for entity in pred_entities],
            labels=[entity.labels for entity in pred_entities],
        )

    def forward(
        self,
        inputs: T_OTXBatchDataEntity,
    ) -> T_OTXBatchPredEntity | OTXBatchLossEntity:
        """Model forward function."""
        # If customize_inputs is overrided
        if isinstance(inputs, (TileBatchDetDataEntity, TileBatchInstSegDataEntity)):
            return self.unpack_tiles(inputs)

        outputs = (
            self.model(**self._customize_inputs(inputs))
            if self._customize_inputs != OTXModel._customize_inputs
            else self.model(inputs)
        )

        return (
            self._customize_outputs(outputs, inputs)
            if self._customize_outputs != OTXModel._customize_outputs
            else outputs
        )
