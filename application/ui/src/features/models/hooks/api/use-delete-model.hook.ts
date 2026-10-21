// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '@/api';
import { useQueryClient } from '@tanstack/react-query';
import { PROJECTS_QUERY_KEY } from 'hooks/api/project.hook';

import { getQueryKey } from '../../../../query-client/query-client';

export const useDeleteModel = () => {
    const queryClient = useQueryClient();

    return $api.useMutation('delete', '/api/projects/{project_id}/models/{model_id}', {
        onSuccess: (
            _,
            {
                params: {
                    path: { project_id, model_id },
                },
            }
        ) => {
            return Promise.all([
                queryClient.invalidateQueries({
                    queryKey: getQueryKey([
                        'get',
                        '/api/projects/{project_id}/models',
                        { params: { path: { project_id } } },
                    ]),
                }),
                queryClient.invalidateQueries({
                    queryKey: getQueryKey([
                        'get',
                        '/api/projects/{project_id}/models/{model_id}',
                        { params: { path: { project_id, model_id } } },
                    ]),
                }),
                queryClient.invalidateQueries({ queryKey: PROJECTS_QUERY_KEY }),
            ]);
        },
    });
};
