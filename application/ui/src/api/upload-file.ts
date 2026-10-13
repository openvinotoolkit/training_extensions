// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { i18n } from '@/i18n';

import { fetchClient } from './client';
import type { MediaDTO, SourceMediaUpload, StagedDataset } from './shared-types';
import { createAbortError, discardUpload, transferFile, type TransferOptions } from './tus-upload';

// openapi-fetch resolves with `{ data, error }` rather than rejecting, and its result union does
// not narrow `data` from the `error` check, so both have to be tested before returning. Each
// caller supplies its own translated, operation-specific fallback message for the rare case where
// both `data` and `error` are missing.
const unwrap = <T>(result: { data?: T; error?: unknown }, fallbackMessage: string): T => {
    if (result.error !== undefined || result.data === undefined) {
        throw result.error ?? new Error(fallbackMessage);
    }

    return result.data;
};

/**
 * Transfers a file through the resumable upload API, then hands the completed upload to `consume`,
 * which calls one of the `:from-upload` endpoints.
 *
 * Cancelling is only possible during the transfer: once the upload is being consumed the server
 * creates the resource regardless. If consumption fails, the upload is discarded right away rather
 * than lingering on the server until it expires.
 */
const uploadResumable = async <T>(
    file: File,
    consume: (uploadId: string) => Promise<T>,
    options: TransferOptions = {}
): Promise<T> => {
    const uploadId = await transferFile(file, options);

    if (options.signal?.aborted) {
        await discardUpload(uploadId);
        throw createAbortError();
    }

    try {
        return await consume(uploadId);
    } catch (error) {
        await discardUpload(uploadId);
        throw error;
    }
};

/** Uploads a single image or video into a project's dataset through a resumable transfer. */
export const uploadDatasetMedia = (projectId: string, file: File, options?: TransferOptions): Promise<MediaDTO> =>
    uploadResumable(
        file,
        async (uploadId) =>
            unwrap(
                await fetchClient.POST('/api/projects/{project_id}/dataset/media:from-upload', {
                    params: { path: { project_id: projectId } },
                    body: { upload_id: uploadId },
                }),
                i18n.t('dataset.upload.genericError')
            ),
        options
    );

/** Uploads a dataset archive (.zip) to the import staging area through a resumable transfer. */
export const uploadDatasetArchive = (file: File, options?: TransferOptions): Promise<StagedDataset> =>
    uploadResumable(
        file,
        async (uploadId) =>
            unwrap(
                await fetchClient.POST('/api/staged_datasets:from-upload', { body: { upload_id: uploadId } }),
                i18n.t('dataset.import.prepareError')
            ),
        options
    );

/** Uploads a video file to be used as an inference pipeline source through a resumable transfer. */
export const uploadSourceVideo = (file: File, options?: TransferOptions): Promise<SourceMediaUpload> =>
    uploadResumable(
        file,
        async (uploadId) =>
            unwrap(
                await fetchClient.POST('/api/sources/media:from-upload', { body: { upload_id: uploadId } }),
                i18n.t('inference.sources.uploadError')
            ),
        options
    );

/** Deletes an uploaded source video that is not referenced by any source. */
export const deleteSourceVideo = async (sourceMediaId: string): Promise<void> => {
    const endpoint = '/api/sources/media/{source_media_id}';
    const result = await fetchClient.DELETE(endpoint, {
        params: { path: { source_media_id: sourceMediaId } },
    });

    unwrap(result, i18n.t('inference.sources.deleteVideoError'));
};
