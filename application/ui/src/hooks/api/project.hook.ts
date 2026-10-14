// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '@/api';
import { useQueryClient, useSuspenseQuery } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { getQueryKey } from '../../query-client/query-client';

// Partial key: matches the projects list query regardless of its query params.
export const PROJECTS_QUERY_KEY = getQueryKey(['get', '/api/projects', { params: {} }]);

export const useProjects = () => {
    return $api.useSuspenseQuery('get', '/api/projects', {
        params: { query: { include_model_counts: true } },
    });
};

export const getProjectQueryOptions = (projectId: string) => {
    return $api.queryOptions('get', '/api/projects/{project_id}', {
        params: { path: { project_id: projectId } },
    });
};

export const useProject = () => {
    const projectId = useProjectIdentifier();

    return useSuspenseQuery(getProjectQueryOptions(projectId));
};

export const useCreateProject = () => {
    return $api.useMutation('post', '/api/projects', {
        meta: { invalidateQueries: [PROJECTS_QUERY_KEY] },
    });
};

export const usePatchProject = () => {
    return $api.useMutation('patch', '/api/projects/{project_id}', {
        meta: { invalidateQueries: [PROJECTS_QUERY_KEY] },
    });
};

export const useDeleteProject = () => {
    const queryClient = useQueryClient();

    return $api.useMutation('delete', '/api/projects/{project_id}', {
        meta: { invalidateQueries: [PROJECTS_QUERY_KEY] },
        onSuccess: async (
            _,
            {
                params: {
                    path: { project_id },
                },
            }
        ) => {
            return queryClient.removeQueries({
                queryKey: ['get', '/api/projects/{project_id}', { params: { path: { project_id } } }],
            });
        },
    });
};
