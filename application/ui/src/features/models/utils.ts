// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Model } from '@/api/types';

export type SelectableModel = {
    modelVariantId: string;
    name: string;
    modelId: string;
    optimalConfidenceThreshold: number | null;
};

export const getModelIdentifierPayload = (model: SelectableModel): { model_id: string; model_variant_id: string } => ({
    model_id: model.modelId,
    model_variant_id: model.modelVariantId,
});

export const distributeByLargestRemainder = (values: number[], total: number): number[] => {
    const sum = values.reduce((acc, value) => acc + value, 0);

    if (sum <= 0 || total <= 0) {
        return values.map(() => 0);
    }

    const exactShares = values.map((value) => (value / sum) * total);
    const flooredShares = exactShares.map((share) => Math.floor(share));
    let remainder = total - flooredShares.reduce((acc, value) => acc + value, 0);

    const indicesByRemainder = exactShares
        .map((share, index) => ({ index, fractional: share - Math.floor(share) }))
        .sort((a, b) => b.fractional - a.fractional);

    const result = [...flooredShares];
    for (const { index } of indicesByRemainder) {
        if (remainder <= 0) break;
        result[index] += 1;
        remainder -= 1;
    }

    return result;
};

export const getAllModelsWithOpenVINOVariants = (models: Model[]): SelectableModel[] => {
    return models.flatMap((model) =>
        model.variants
            .filter((variant) => variant.format === 'openvino' && model.files_deleted === false)
            .map((variant): SelectableModel => ({
                modelVariantId: variant.id,
                modelId: model.id,
                name: `${model.name} [${variant.precision.toUpperCase()}]`,
                optimalConfidenceThreshold: variant.optimal_confidence_threshold ?? null,
            }))
    );
};

export const isUltralyticsModel = (identifier: string): boolean => {
    return /yolo\d+-/.test(identifier.toLocaleLowerCase());
};

export const isEdgeCrafterModel = (identifier: string): boolean => {
    return /edgecrafter-/.test(identifier.toLocaleLowerCase());
};
