// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Synthetic card returned by the backend that opens the timm backbone selector.
// It is never trainable itself - a concrete `image-classification-timm-<name>` id is resolved instead.
export const TIMM_MODEL_ARCHITECTURE_ID = 'image-classification-timm';

export const isTimmModelArchitecture = (modelArchitectureId: string | null): boolean =>
    modelArchitectureId === TIMM_MODEL_ARCHITECTURE_ID;
