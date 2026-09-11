// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { i18n } from '@/i18n';

import { fetchClient } from './client';
import type { MediaDTO, SourceMediaUpload, StagedDataset } from './shared-types';

/**
 * Wraps a file in the `multipart/form-data` body every Geti upload endpoint expects.
 *
 * The generic `TBody` is inferred from the endpoint's expected request body: the OpenAPI
 * generator describes a multipart body as an object with a `file` field, but openapi-fetch has
 * to receive the FormData instance itself. This is the only place that discrepancy is handled.
 */
const fileBody = <TBody>(file: File): NonNullable<TBody> => {
    const body = new FormData();
    body.append('file', file);

    return body as unknown as NonNullable<TBody>;
};

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

/** Uploads a single image or video into a project's dataset. */
export const uploadDatasetMedia = async (projectId: string, file: File): Promise<MediaDTO> => {
    const endpoint = '/api/projects/{project_id}/dataset/media';
    const result = await fetchClient.POST(endpoint, {
        params: { path: { project_id: projectId } },
        body: fileBody(file),
    });

    return unwrap(result, i18n.t('dataset.upload.genericError'));
};

/** Uploads a dataset archive (.zip) to the import staging area. */
export const uploadDatasetArchive = async (file: File): Promise<StagedDataset> => {
    const endpoint = '/api/staged_datasets';
    const result = await fetchClient.POST(endpoint, { body: fileBody(file) });

    return unwrap(result, i18n.t('dataset.import.prepareError'));
};

/** Uploads a video file to be used as an inference pipeline source. */
export const uploadSourceVideo = async (file: File): Promise<SourceMediaUpload> => {
    const endpoint = '/api/sources/media';
    const result = await fetchClient.POST(endpoint, { body: fileBody(file) });

    return unwrap(result, i18n.t('inference.sources.uploadError'));
};
