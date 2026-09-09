// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '@/api';
import { usePrefetchQuery, useQuery, useQueryClient, useSuspenseQuery } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { getQueryKey } from '../../query-client/query-client';

const getPipelineQueryOptions = (projectId: string) => {
    return $api.queryOptions('get', '/api/projects/{project_id}/pipeline', {
        params: {
            path: {
                project_id: projectId,
            },
        },
    });
};

export const usePipeline = () => {
    const projectId = useProjectIdentifier();

    return useSuspenseQuery(getPipelineQueryOptions(projectId));
};

export const usePrefetchPipeline = () => {
    const projectId = useProjectIdentifier();

    return usePrefetchQuery(getPipelineQueryOptions(projectId));
};

export const useProjectPipeline = (projectId: string) => {
    return useQuery({ ...getPipelineQueryOptions(projectId), retry: false });
};

const POLLING_INTERVAL = 5000;
export const usePipelineMetrics = () => {
    const projectId = useProjectIdentifier();

    return $api.useQuery(
        'get',
        '/api/projects/{project_id}/pipeline/metrics',
        {
            params: { path: { project_id: projectId } },
        },
        {
            refetchInterval: (query) => (query.state.status === 'success' ? POLLING_INTERVAL : false),
            retry: false,
        }
    );
};

export const usePipelineHealth = () => {
    const projectId = useProjectIdentifier();

    return $api.useQuery(
        'get',
        '/api/projects/{project_id}/pipeline/health',
        {
            params: { path: { project_id: projectId } },
        },
        {
            refetchInterval: POLLING_INTERVAL,
            retry: false,
        }
    );
};

export const usePatchPipeline = () => {
    const queryClient = useQueryClient();

    return $api.useMutation('patch', '/api/projects/{project_id}/pipeline', {
        onSuccess: (
            _,
            {
                params: {
                    path: { project_id },
                },
            }
        ) => {
            return queryClient.invalidateQueries({
                queryKey: getQueryKey([
                    'get',
                    '/api/projects/{project_id}/pipeline',
                    { params: { path: { project_id } } },
                ]),
            });
        },
    });
};

export const useEnablePipeline = () => {
    const queryClient = useQueryClient();

    return $api.useMutation('post', '/api/projects/{project_id}/pipeline:enable', {
        onSuccess: (
            _,
            {
                params: {
                    path: { project_id },
                },
            }
        ) => {
            return Promise.all([
                queryClient.invalidateQueries({
                    queryKey: getQueryKey([
                        'get',
                        '/api/projects/{project_id}/pipeline',
                        { params: { path: { project_id } } },
                    ]),
                }),
                queryClient.invalidateQueries({
                    queryKey: getQueryKey(['get', '/api/projects']),
                }),
            ]);
        },
    });
};

export const useDisablePipeline = () => {
    const queryClient = useQueryClient();

    return $api.useMutation('post', '/api/projects/{project_id}/pipeline:disable', {
        onSuccess: (
            _,
            {
                params: {
                    path: { project_id },
                },
            }
        ) => {
            return Promise.all([
                queryClient.invalidateQueries({
                    queryKey: getQueryKey([
                        'get',
                        '/api/projects/{project_id}/pipeline',
                        { params: { path: { project_id } } },
                    ]),
                }),
                queryClient.invalidateQueries({
                    queryKey: getQueryKey(['get', '/api/projects']),
                }),
            ]);
        },
    });
};

export const useCapturePipelineFrame = () => {
    return $api.useMutation('post', '/api/projects/{project_id}/pipeline:capture');
};

export const useConnectSourceToPipeline = () => {
    const project_id = useProjectIdentifier();
    const pipeline = usePatchPipeline();

    return (source_id: string) => pipeline.mutateAsync({ params: { path: { project_id } }, body: { source_id } });
};

export const useConnectSinkToPipeline = () => {
    const project_id = useProjectIdentifier();
    const pipeline = usePatchPipeline();

    return (sink_id: string) => pipeline.mutateAsync({ params: { path: { project_id } }, body: { sink_id } });
};
