// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '@/api';
import { useSuspenseQuery } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

export const datasetViewsQueryOptions = (projectId: string) =>
    $api.queryOptions(
        'get',
        '/api/projects/{project_id}/dataset/views',
        {
            params: {
                path: {
                    project_id: projectId,
                },
            },
        },
        { staleTime: 1000 * 60 }
    );

export const useDatasetViewsQuery = () => {
    const projectId = useProjectIdentifier();
    return useSuspenseQuery(datasetViewsQueryOptions(projectId));
};
