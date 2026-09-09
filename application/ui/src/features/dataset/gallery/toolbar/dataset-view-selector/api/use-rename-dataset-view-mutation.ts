// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '@/api';
import { useQueryClient } from '@tanstack/react-query';

import { getQueryKey } from '../../../../../../query-client/query-client';
import { datasetViewsQueryOptions } from './use-dataset-views';

export const useRenameDatasetViewMutation = () => {
    const queryClient = useQueryClient();

    return $api.useMutation('patch', '/api/projects/{project_id}/dataset/views/{dataset_view_id}', {
        onSuccess: ({ id, project_id }) => {
            return Promise.all([
                queryClient.invalidateQueries({
                    queryKey: getQueryKey([
                        'get',
                        '/api/projects/{project_id}/dataset/views/{dataset_view_id}',
                        { params: { path: { project_id, dataset_view_id: id } } },
                    ]),
                }),
                queryClient.invalidateQueries({
                    queryKey: datasetViewsQueryOptions(project_id).queryKey,
                }),
            ]);
        },
    });
};
