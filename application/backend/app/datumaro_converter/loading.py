# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Load stored Datumaro datasets and upgrade schemas written by older application versions."""

import copy
from pathlib import Path
from typing import Any

from datumaro.experimental import Dataset, Sample
from datumaro.experimental.export_import import import_dataset as dm_import_dataset
from loguru import logger

from .domain import (
    DetectionImportExportSample,
    DetectionTrainingSample,
    InstanceSegmentationImportExportSample,
    InstanceSegmentationTrainingSample,
    MulticlassClassificationImportExportSample,
    MulticlassClassificationTrainingSample,
    MultilabelClassificationImportExportSample,
    MultilabelClassificationTrainingSample,
)

# Samples whose ``label`` field was widened from ``UInt8`` to ``UInt16`` to support more than 256 labels.
# Datumaro resolves the sample type of a stored dataset by exact schema match, so datasets saved with the
# previous ``UInt8`` label (dataset revisions, exported archives) would otherwise load as the base ``Sample``.
_WIDENED_LABEL_SAMPLES: tuple[type[Sample], ...] = (
    MulticlassClassificationTrainingSample,
    MulticlassClassificationImportExportSample,
    MultilabelClassificationTrainingSample,
    MultilabelClassificationImportExportSample,
    DetectionTrainingSample,
    DetectionImportExportSample,
    InstanceSegmentationTrainingSample,
    InstanceSegmentationImportExportSample,
)
_LEGACY_LABEL_DTYPE = "UInt8"


def _legacy_schema_attributes(sample_type: type[Sample]) -> dict[str, Any]:
    """Return the serialized schema attributes of ``sample_type`` as written with the legacy ``UInt8`` label."""
    attributes = copy.deepcopy(sample_type.infer_schema().to_dict()["attributes"])
    attributes["label"]["field"]["dtype"] = _LEGACY_LABEL_DTYPE
    return attributes


def upgrade_legacy_dataset(dataset: Dataset) -> Dataset:
    """
    Convert a dataset stored with a legacy ``UInt8`` label schema to the current sample type.

    Datasets that were matched to a registered sample type, or whose schema is not a known legacy schema,
    are returned unchanged. The conversion widens ``UInt8`` labels to ``UInt16`` and is lossless.

    Args:
        dataset: Dataset loaded from storage.

    Returns:
        The dataset typed with the current sample class, or the input dataset if no upgrade applies.
    """
    if dataset.dtype is not Sample:
        return dataset
    stored_attributes = dataset.schema.to_dict().get("attributes", {})
    for sample_type in _WIDENED_LABEL_SAMPLES:
        if stored_attributes == _legacy_schema_attributes(sample_type):
            logger.info("Upgrading dataset with legacy UInt8 label schema to {}", sample_type.__name__)
            return dataset.convert_to_schema(sample_type)
    return dataset


def import_dataset(input_path: str | Path, extract_dir: str | Path | None = None) -> Dataset:
    """
    Import a stored Datumaro dataset, upgrading legacy ``UInt8`` label schemas.

    Args:
        input_path: Path to the dataset directory or archive.
        extract_dir: Optional directory to extract the archive to.

    Returns:
        The imported dataset.
    """
    return upgrade_legacy_dataset(dm_import_dataset(input_path, extract_dir=extract_dir))
