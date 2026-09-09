// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { API_BASE_URL, fetchClient } from '@/api';
import { useMutation, useQuery } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { downloadFile } from '../../../../platform/download-file';
import { getQueryKey } from '../../../../query-client/query-client';
import { assertIsNotNullable } from '../../../../shared/util';
import { type LogEntry } from '../log-types';
import { parseLogLine } from '../log-utils';

const fetchModelLogs = async (projectId: string, modelId: string): Promise<LogEntry[]> => {
    const { data, error, response } = await fetchClient.GET('/api/projects/{project_id}/models/{model_id}/logs', {
        params: { path: { project_id: projectId, model_id: modelId } },
        parseAs: 'text',
    });

    if (error) {
        throw new Error(`Failed to fetch model logs: ${response.status} ${response.statusText}`);
    }

    const text = data ?? '';

    return text
        .split('\n')
        .filter((line) => line.trim())
        .map((line) => parseLogLine(line))
        .filter((entry): entry is LogEntry => entry !== null);
};

export const useModelLogs = (modelId: string | undefined) => {
    const projectId = useProjectIdentifier();

    return useQuery({
        queryKey: getQueryKey([
            'get',
            '/api/projects/{project_id}/models/{model_id}/logs',
            { params: { path: { project_id: projectId, model_id: modelId } } },
        ]),
        queryFn: () => {
            assertIsNotNullable(modelId);

            return fetchModelLogs(projectId, modelId);
        },
        enabled: !!modelId,
        staleTime: Infinity, // Completed/failed model logs don't change
    });
};

const downloadModelLogsFile = (projectId: string, modelId: string) => {
    const url = `${API_BASE_URL}/api/projects/${projectId}/models/${modelId}/logs`;
    downloadFile(url, `training-logs-${modelId}.log`, 'Training logs download started');
};

export const useDownloadModelLogs = (modelId: string) => {
    const projectId = useProjectIdentifier();

    const mutation = useMutation({
        mutationFn: async () => {
            assertIsNotNullable(modelId);

            await downloadModelLogsFile(projectId, modelId);
        },
    });

    return {
        downloadModelLogs: () => mutation.mutate(),
        isDownloading: mutation.isPending,
    };
};
