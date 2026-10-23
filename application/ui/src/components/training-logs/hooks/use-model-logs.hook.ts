// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fetchClient } from '@/api';
import { useTranslation, type TranslateFn } from '@/i18n';
import { useQuery } from '@tanstack/react-query';

import { getQueryKey } from '../../../query-client/query-client';
import { assertIsNotNullable } from '../../../shared/util';
import { type LogEntry } from '../log-types';
import { parseLogLine } from '../log-utils';

const fetchModelLogs = async (t: TranslateFn, projectId: string, modelId: string): Promise<LogEntry[]> => {
    const { data, error, response } = await fetchClient.GET('/api/projects/{project_id}/models/{model_id}/logs', {
        params: { path: { project_id: projectId, model_id: modelId } },
        parseAs: 'text',
    });

    if (error) {
        throw new Error(
            t('models.training.logs.fetchError', { status: response.status, statusText: response.statusText })
        );
    }

    const text = data ?? '';

    return text
        .split('\n')
        .filter((line) => line.trim())
        .map((line) => parseLogLine(line))
        .filter((entry): entry is LogEntry => entry !== null);
};

export const useModelLogs = (projectId: string, modelId: string | undefined) => {
    const { t } = useTranslation();

    return useQuery({
        queryKey: getQueryKey([
            'get',
            '/api/projects/{project_id}/models/{model_id}/logs',
            { params: { path: { project_id: projectId, model_id: modelId ?? '' } } },
        ]),
        queryFn: () => {
            assertIsNotNullable(modelId);

            return fetchModelLogs(t, projectId, modelId);
        },
        enabled: !!modelId,
        staleTime: Infinity,
    });
};
