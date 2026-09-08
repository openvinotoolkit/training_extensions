// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { uploadDatasetMedia } from '@/api';
import type { MediaDTO } from '@/api/types';
import { useQueryClient } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isFunction } from 'lodash-es';

import { getErrorMessage, getQueryKey } from '../../../query-client/query-client';
import { useUploadProgress } from '../hooks/use-display-upload-progress';

export const MEDIA_UPLOAD_CONCURRENCY = 10;

type UploadTask<T> = () => Promise<T>;

// Runs ${batchSize} promises at a time until all promises have been executed,
// returning an array of settled results.
const executeInBatches = async <T>(
    uploadTasks: UploadTask<T>[],
    batchSize: number,
    onBatchCompleted?: (batchResults: PromiseSettledResult<T>[]) => Promise<void>
): Promise<PromiseSettledResult<T>[]> => {
    const settledResults: PromiseSettledResult<T>[] = [];

    for (let index = 0; index < uploadTasks.length; index += batchSize) {
        const batchPromises = uploadTasks.slice(index, index + batchSize).map((task) => task());
        const batchResults = await Promise.allSettled(batchPromises);

        settledResults.push(...batchResults);

        if (isFunction(onBatchCompleted)) {
            await onBatchCompleted(batchResults);
        }
    }

    return settledResults;
};

const getFulfilledValues = <T>(results: PromiseSettledResult<T>[]): T[] =>
    results
        .filter((result): result is PromiseFulfilledResult<T> => result.status === 'fulfilled')
        .map((result) => result.value);

export const useMediaUpload = () => {
    const projectId = useProjectIdentifier();
    const queryClient = useQueryClient();
    const {
        uploadProgress,
        startUploadProgress,
        setItemUploading,
        setItemUploaded,
        setItemFailed,
        finishUploadProgress,
    } = useUploadProgress();

    const buildUploadTask = (file: File, itemId: string): UploadTask<MediaDTO> => {
        return async () => {
            setItemUploading(itemId);

            try {
                const result = await uploadDatasetMedia(projectId, file);
                setItemUploaded(itemId);

                return result;
            } catch (error) {
                setItemFailed(itemId, getErrorMessage(error));
                throw error;
            }
        };
    };

    const invalidateMediaQuery = () => {
        return Promise.all([
            queryClient.invalidateQueries({
                queryKey: getQueryKey([
                    'get',
                    '/api/projects/{project_id}/dataset/media',
                    { params: { path: { project_id: projectId } } },
                ]),
            }),
            queryClient.invalidateQueries({
                queryKey: getQueryKey([
                    'get',
                    '/api/projects/{project_id}/dataset/statistics',
                    { params: { path: { project_id: projectId } } },
                ]),
            }),
        ]);
    };

    // Processes files with batched concurrency, returning all successfully uploaded media items
    const processUploadBatch = async (files: File[]): Promise<MediaDTO[]> => {
        if (files.length === 0) {
            return [];
        }

        const itemIds = startUploadProgress(files);

        try {
            const onBatchCompleted = async () => {
                await invalidateMediaQuery();
            };

            const uploadTasks = files.map((file, index) => buildUploadTask(file, itemIds[index]));
            const allResults = await executeInBatches(uploadTasks, MEDIA_UPLOAD_CONCURRENCY, onBatchCompleted);

            finishUploadProgress();

            return getFulfilledValues(allResults);
        } catch (_error) {
            finishUploadProgress();
            return [];
        }
    };

    // Starts the upload process directly, returning all successfully uploaded media items
    const uploadMedia = async (files: File[]): Promise<MediaDTO[]> => {
        if (files.length === 0) {
            return [];
        }

        return processUploadBatch(files);
    };

    return {
        uploadMedia,
        uploadProgress,
    };
};
