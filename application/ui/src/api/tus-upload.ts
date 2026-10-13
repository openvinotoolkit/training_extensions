// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { i18n } from '@/i18n';
import { isObject } from 'lodash-es';
import { DetailedError, Upload, type PreviousUpload } from 'tus-js-client';

import { API_BASE_URL, fetchClient } from './client';

export const TUS_VERSION = '1.0.0';

// Chunking bounds how much a dropped connection costs and keeps each request under reverse-proxy
// body size limits, while staying large enough for the per-request overhead to be negligible.
export const TUS_CHUNK_SIZE = 16 * 1024 * 1024;

// Before each retry tus-js-client asks the server for the committed offset (HEAD), so a transient
// network failure only costs the chunk that was in flight.
const RETRY_DELAYS_MS = [0, 1_000, 3_000, 5_000, 10_000, 20_000];

const UPLOAD_URL_PATTERN = /\/api\/uploads\/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$/i;

export type TransferOptions = {
    /** Called whenever more bytes of the file have been sent to the server. */
    onProgress?: (bytesSent: number) => void;
    /** Aborting cancels the transfer and discards whatever the server has received so far. */
    signal?: AbortSignal;
};

export const createAbortError = (): DOMException => new DOMException('The upload was cancelled.', 'AbortError');

export const isAbortError = (error: unknown): boolean =>
    isObject(error) && 'name' in error && error.name === 'AbortError';

const getUploadsEndpoint = (): string => `${(API_BASE_URL || window.location.origin).replace(/\/$/, '')}/api/uploads`;

const getUploadId = (uploadUrl: string | null): string => {
    const uploadId = uploadUrl?.match(UPLOAD_URL_PATTERN)?.[1];

    if (uploadId === undefined) {
        throw new Error(i18n.t('application.notifications.uploadTransferError'));
    }

    return uploadId;
};

// tus-js-client errors embed the raw request/response dump in their message; surface the server's
// `detail` instead (e.g. "Upload exceeds the maximum size") or a generic, translated message.
const toTransferError = (error: Error): Error => {
    if (!(error instanceof DetailedError)) {
        return error;
    }

    if (error.originalResponse === null) {
        return new Error(i18n.t('application.notifications.networkError'));
    }

    try {
        const { detail } = JSON.parse(error.originalResponse.getBody()) as { detail?: unknown };

        if (typeof detail === 'string') {
            return new Error(detail);
        }
    } catch {
        // Not a JSON error body; fall through to the generic message.
    }

    return new Error(i18n.t('application.notifications.uploadTransferError'));
};

const getMostRecentUpload = (previousUploads: PreviousUpload[]): PreviousUpload | undefined =>
    previousUploads.toSorted((a, b) => b.creationTime.localeCompare(a.creationTime))[0];

/**
 * Transfers a file to the resumable upload API (`/api/uploads`, TUS 1.0.0) and resolves with the
 * id of the completed upload, ready to be handed to one of the `:from-upload` endpoints.
 *
 * Interrupted transfers of the same file (same name, type, size and modification time) resume from
 * the offset the server has committed, including across page reloads.
 */
export const transferFile = (file: File, { onProgress, signal }: TransferOptions = {}): Promise<string> =>
    new Promise<string>((resolve, reject) => {
        if (signal?.aborted) {
            reject(createAbortError());
            return;
        }

        let isSettled = false;
        const settle = (callback: () => void): void => {
            if (!isSettled) {
                isSettled = true;
                signal?.removeEventListener('abort', onAbort);
                callback();
            }
        };

        const upload = new Upload(file, {
            endpoint: getUploadsEndpoint(),
            chunkSize: TUS_CHUNK_SIZE,
            retryDelays: RETRY_DELAYS_MS,
            metadata: file.type ? { filename: file.name, filetype: file.type } : { filename: file.name },
            storeFingerprintForResuming: true,
            removeFingerprintOnSuccess: true,
            onProgress: (bytesSent) => onProgress?.(bytesSent),
            onSuccess: () =>
                settle(() => {
                    try {
                        resolve(getUploadId(upload.url));
                    } catch (error) {
                        reject(error);
                    }
                }),
            onError: (error) => settle(() => reject(toTransferError(error))),
        });

        const onAbort = (): void =>
            settle(() => {
                // Terminating (TUS DELETE) discards the partial bytes on the server and forgets the
                // stored resume point, so a cancelled upload leaves nothing behind.
                upload.abort(true).catch(() => undefined);
                reject(createAbortError());
            });

        signal?.addEventListener('abort', onAbort, { once: true });

        upload
            .findPreviousUploads()
            .catch((): PreviousUpload[] => [])
            .then((previousUploads) => {
                if (isSettled) {
                    return;
                }

                const previousUpload = getMostRecentUpload(previousUploads);

                if (previousUpload !== undefined) {
                    upload.resumeFromPreviousUpload(previousUpload);
                }

                upload.start();
            });
    });

/** Discards an upload on the server (TUS termination). Failures are ignored: the upload expires anyway. */
export const discardUpload = async (uploadId: string): Promise<void> => {
    try {
        await fetchClient.DELETE('/api/uploads/{upload_id}', {
            params: { path: { upload_id: uploadId }, header: { 'Tus-Resumable': TUS_VERSION } },
        });
    } catch {
        // The server garbage-collects expired uploads, so there is nothing else to do.
    }
};
