# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from otx.core.config.explain import ExplainConfig


def get_processed_saliency_maps(raw_saliency_maps: list, explain_config: ExplainConfig | None, predictions: list | None) -> list | None:
    # Implement filtering and post-processing <- TODO(negvet)
    return raw_saliency_maps
