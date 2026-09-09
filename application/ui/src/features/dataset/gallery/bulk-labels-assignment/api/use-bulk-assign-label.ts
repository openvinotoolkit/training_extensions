// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '@/api';
import { toast } from '@/components/toast/toast.component';
import { useQueryClient } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isEmpty, partition } from 'lodash-es';

import { getQueryKey } from '../../../../../query-client/query-client';
import { filterOutEmptyLabels } from '../../../../../shared/annotator/labels';

export const useBulkAssignLabel = () => {
    const projectId = useProjectIdentifier();
    const queryClient = useQueryClient();
    const mutation = $api.useMutation('post', '/api/projects/{project_id}/dataset/media/{media_id}/annotations');

    const invalidateQueries = () => {
        queryClient.invalidateQueries({
            queryKey: getQueryKey([
                'get',
                '/api/projects/{project_id}/dataset/items',
                { params: { path: { project_id: projectId } } },
            ]),
        });
        queryClient.invalidateQueries({
            queryKey: getQueryKey([
                'get',
                '/api/projects/{project_id}/dataset/media',
                {
                    params: {
                        path: { project_id: projectId },
                    },
                },
            ]),
        });
    };

    const assignLabel = async (mediaId: string, labelIds: string[]) => {
        const labelsWithoutEmptyLabel = filterOutEmptyLabels(labelIds.map((id) => ({ id })));

        return mutation.mutateAsync({
            params: {
                path: {
                    project_id: projectId,
                    media_id: mediaId,
                },
            },
            body: {
                annotations: isEmpty(labelsWithoutEmptyLabel)
                    ? []
                    : [{ shape: { type: 'full_image' }, labels: labelsWithoutEmptyLabel }],
            },
        });
    };

    const bulkAssignLabel = async (mediaIds: string[], labelIds: string[]) => {
        const result = await Promise.allSettled(mediaIds.map((mediaId) => assignLabel(mediaId, labelIds)));

        const [successfulMediaItems, failedMediaItems] = partition(result, ({ status }) => status === 'fulfilled');

        if (failedMediaItems.length === 0) {
            toast({
                type: 'success',
                message: `Successfully assigned label(s) to all ${successfulMediaItems.length} image(s)`,
            });
        } else if (successfulMediaItems.length === 0) {
            toast({
                type: 'error',
                message: `Failed to assign label(s) to all ${failedMediaItems.length} image(s)`,
            });
        } else {
            toast({
                type: 'info',
                message:
                    `Assigned label(s) to ${successfulMediaItems.length} of ${mediaIds.length} image(s) ` +
                    `(${failedMediaItems.length} failed)`,
            });
        }

        invalidateQueries();
    };

    return {
        mutate: bulkAssignLabel,
        isPending: mutation.isPending,
    };
};
