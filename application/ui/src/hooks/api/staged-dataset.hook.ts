// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '@/api';

import { isNonEmptyString } from '../../shared/util';
import { isInvalidStagedFile } from './util';

type useDeleteStagedDatasetProps = {
    stagedDatasetId: string | null | undefined;
    onError?: (error: unknown) => void;
    onSuccess?: () => void;
    deleteEntry?: () => void;
};

export const useStagedDataset = (stagedDatasetId: string | null | undefined) => {
    return $api.useQuery(
        'get',
        '/api/staged_datasets/{staged_dataset_id}',
        {
            params: { path: { staged_dataset_id: stagedDatasetId ?? '' } },
        },
        { enabled: isNonEmptyString(stagedDatasetId) }
    );
};

export const useStagedDatasetSuspense = (stagedDatasetId: string) => {
    return $api.useSuspenseQuery('get', '/api/staged_datasets/{staged_dataset_id}', {
        params: { path: { staged_dataset_id: stagedDatasetId } },
    });
};

export const useDeleteStagedDataset = ({
    stagedDatasetId,
    onError,
    onSuccess,
    deleteEntry,
}: useDeleteStagedDatasetProps) => {
    const params = { params: { path: { staged_dataset_id: stagedDatasetId ?? '' } } };
    const hasStagedDatasetId = isNonEmptyString(stagedDatasetId);

    const deleteMutation = $api.useMutation('delete', '/api/staged_datasets/{staged_dataset_id}', {
        meta: { error: { notify: (error: unknown) => !isInvalidStagedFile(error) } },
        onSuccess: () => {
            deleteEntry?.();
            onSuccess?.();
        },
        onError: (error) => {
            isInvalidStagedFile(error) && deleteEntry?.();
            onError?.(error);
        },
    });

    return {
        ...deleteMutation,
        mutate: () => {
            if (hasStagedDatasetId) {
                deleteMutation.mutate(params);
            }
        },
        mutateAsync: () => (hasStagedDatasetId ? deleteMutation.mutateAsync(params) : Promise.resolve(undefined)),
    };
};
