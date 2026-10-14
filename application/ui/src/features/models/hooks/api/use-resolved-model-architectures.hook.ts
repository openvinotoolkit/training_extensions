// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '@/api';
import type { Model, ModelArchitecture, ModelArchitectureWithPerformanceCategory } from '@/api/types';
import { useQueries } from '@tanstack/react-query';

import { useGetTaskModelArchitectures } from './use-get-model-architectures.hook';

const getArchitectureByIdQueryOptions = (architectureId: string) =>
    $api.queryOptions(
        'get',
        '/api/model_architectures/{architecture_id}',
        { params: { path: { architecture_id: architectureId } } },
        // The architecture catalog is static; resolved entries never change
        { staleTime: Infinity }
    );

/**
 * Extends the task's model architectures with on-demand resolution of architecture IDs
 * that are not part of the static list.
 *
 * Trained timm models carry a concrete `image-classification-timm-<name>` architecture
 * ID which never appears in the architectures list endpoint (only the synthetic
 * "Other models (TIMM)" selector card does). Without resolution those models would show
 * their raw architecture ID and no license in the models table.
 */
export const useResolvedModelArchitectures = (models: Model[]): ModelArchitectureWithPerformanceCategory[] => {
    const { modelArchitectures } = useGetTaskModelArchitectures();

    const knownIds = new Set(modelArchitectures.map(({ id }) => id));
    const unresolvedIds = [
        ...new Set(models.map(({ architecture }) => architecture).filter((id) => !knownIds.has(id))),
    ];

    const resolvedQueries = useQueries({
        queries: unresolvedIds.map((architectureId) => getArchitectureByIdQueryOptions(architectureId)),
    });

    const resolvedArchitectures = resolvedQueries
        .map(({ data }) => data)
        .filter((architecture): architecture is ModelArchitecture => architecture !== undefined);

    return [...modelArchitectures, ...resolvedArchitectures];
};
