// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '@/api';
import type { TrainingConfiguration } from '@/api/types';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

// openapi-fetch normalizes response types, which widens tuples such as `float_range` values
// to `number[]`. Restore the spec's shape so consumers keep using the generated schema types.
const asTrainingConfiguration = (data: unknown) => data as TrainingConfiguration;

export const useGetModelTrainingConfiguration = (modelId: string | null) => {
    const projectId = useProjectIdentifier();

    return $api.useQuery(
        'get',
        '/api/projects/{project_id}/models/{model_id}/training_configuration',
        {
            params: { path: { project_id: projectId, model_id: modelId ?? '' } },
        },
        {
            enabled: modelId !== null,
            select: asTrainingConfiguration,
        }
    );
};

export const useGetModelArchitectureTrainingConfiguration = ({
    modelArchitectureId,
    modelRevisionId,
}: {
    modelArchitectureId: string | null;
    modelRevisionId: string | null;
}) => {
    const projectId = useProjectIdentifier();

    return $api.useQuery(
        'get',
        '/api/projects/{project_id}/training_configuration',
        {
            params: {
                path: {
                    project_id: projectId,
                },
                query: {
                    model_architecture_id: String(modelArchitectureId),
                },
            },
        },
        {
            enabled: modelArchitectureId !== null && modelRevisionId === null,
            select: asTrainingConfiguration,
        }
    );
};
