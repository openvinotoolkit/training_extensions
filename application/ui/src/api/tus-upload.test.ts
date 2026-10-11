// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedStagedDataset } from 'mocks/mock-staged-dataset';
import { HttpResponse } from 'msw';
import {
    DetailedError,
    type PreviousUpload,
    type HttpResponse as TusHttpResponse,
    type UploadOptions,
} from 'tus-js-client';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { server } from '../msw-node-setup';
import { API_BASE_URL } from './client';
import { isAbortError, TUS_CHUNK_SIZE } from './tus-upload';
import { uploadDatasetArchive } from './upload-file';
import { http } from './utils';

const UPLOAD_ID = '00000000-0000-4000-8000-000000000001';

type FakeUpload = {
    file: File;
    options: UploadOptions;
    url: string | null;
    start: () => void;
    abort: (shouldTerminate?: boolean) => Promise<void>;
    resumeFromPreviousUpload: (previousUpload: PreviousUpload) => void;
};

const tus = vi.hoisted(() => ({
    instances: [] as FakeUpload[],
    previousUploads: [] as PreviousUpload[],
    onStart: (_upload: FakeUpload): void => undefined,
}));

vi.mock('tus-js-client', async (importOriginal) => {
    const actual = await importOriginal<typeof import('tus-js-client')>();

    class Upload implements FakeUpload {
        url: string | null = null;
        start = vi.fn(() => tus.onStart(this));
        abort = vi.fn(async (_shouldTerminate?: boolean) => undefined);
        resumeFromPreviousUpload = vi.fn((previousUpload: PreviousUpload) => {
            this.url = previousUpload.uploadUrl;
        });
        findPreviousUploads = vi.fn(async () => tus.previousUploads);

        constructor(
            public file: File,
            public options: UploadOptions
        ) {
            tus.instances.push(this);
        }
    }

    return { ...actual, Upload };
});

const completeTransfer = (upload: FakeUpload): void => {
    upload.url ??= `http://localhost:3000/api/uploads/${UPLOAD_ID}`;
    upload.options.onProgress?.(upload.file.size / 2, upload.file.size);
    upload.options.onProgress?.(upload.file.size, upload.file.size);
    upload.options.onSuccess?.({ lastResponse: {} as TusHttpResponse });
};

const previousUpload = (uploadUrl: string, creationTime: string): PreviousUpload => ({
    size: 8,
    metadata: {},
    creationTime,
    uploadUrl,
    parallelUploadUrls: null,
    urlStorageKey: `tus::${creationTime}`,
});

const mockEndpoints = () => {
    const consumed: string[] = [];
    const deleted: string[] = [];

    server.use(
        http.post('/api/staged_datasets:from-upload', async ({ request }) => {
            const { upload_id } = await request.json();
            consumed.push(upload_id);

            return HttpResponse.json(getMockedStagedDataset({ id: 'staged-1' }), { status: 201 });
        }),
        http.delete('/api/uploads/{upload_id}', ({ params, request }) => {
            expect(request.headers.get('Tus-Resumable')).toBe('1.0.0');
            deleted.push(params.upload_id);

            return new HttpResponse(null, { status: 204 });
        })
    );

    return { consumed, deleted };
};

describe('resumable upload', () => {
    const file = new File(['12345678'], 'dataset.zip', { type: 'application/zip' });

    beforeEach(() => {
        tus.instances = [];
        tus.previousUploads = [];
        tus.onStart = completeTransfer;
    });

    it('transfers the file with TUS, reports progress and consumes the completed upload', async () => {
        const { consumed, deleted } = mockEndpoints();
        const onProgress = vi.fn();

        await expect(uploadDatasetArchive(file, { onProgress })).resolves.toMatchObject({ id: 'staged-1' });

        const [upload] = tus.instances;
        expect(upload.options).toMatchObject({
            endpoint: `${API_BASE_URL || window.location.origin}/api/uploads`,
            chunkSize: TUS_CHUNK_SIZE,
            metadata: { filename: 'dataset.zip', filetype: 'application/zip' },
            storeFingerprintForResuming: true,
            removeFingerprintOnSuccess: true,
        });
        expect(onProgress).toHaveBeenNthCalledWith(1, 4);
        expect(onProgress).toHaveBeenLastCalledWith(8);
        expect(consumed).toEqual([UPLOAD_ID]);
        expect(deleted).toEqual([]);
    });

    it('does not declare an empty file type', async () => {
        mockEndpoints();

        await uploadDatasetArchive(new File(['x'], 'dataset.zip'));

        expect(tus.instances[0].options.metadata).toEqual({ filename: 'dataset.zip' });
    });

    it('resumes the most recent interrupted transfer of the same file', async () => {
        const { consumed } = mockEndpoints();
        const resumedId = '00000000-0000-4000-8000-000000000002';
        tus.previousUploads = [
            previousUpload('http://localhost:3000/api/uploads/00000000-0000-4000-8000-000000000003', '2026-01-01'),
            previousUpload(`http://localhost:3000/api/uploads/${resumedId}`, '2026-01-02'),
        ];

        await uploadDatasetArchive(file);

        expect(tus.instances[0].resumeFromPreviousUpload).toHaveBeenCalledWith(tus.previousUploads[1]);
        expect(consumed).toEqual([resumedId]);
    });

    it('terminates the upload on the server when cancelled during the transfer', async () => {
        const { consumed } = mockEndpoints();
        const controller = new AbortController();
        tus.onStart = (upload) => {
            upload.url = `http://localhost:3000/api/uploads/${UPLOAD_ID}`;
            upload.options.onProgress?.(2, file.size);
            controller.abort();
        };

        const error = await uploadDatasetArchive(file, { signal: controller.signal }).catch((reason) => reason);

        expect(isAbortError(error)).toBe(true);
        expect(tus.instances[0].abort).toHaveBeenCalledWith(true);
        expect(consumed).toEqual([]);
    });

    it('does not start anything when already cancelled', async () => {
        const controller = new AbortController();
        controller.abort();

        const error = await uploadDatasetArchive(file, { signal: controller.signal }).catch((reason) => reason);

        expect(isAbortError(error)).toBe(true);
        expect(tus.instances).toHaveLength(0);
    });

    it('discards the upload when cancelled after the transfer completed but before consumption', async () => {
        const { consumed, deleted } = mockEndpoints();
        const controller = new AbortController();
        tus.onStart = (upload) => {
            completeTransfer(upload);
            controller.abort();
        };

        const error = await uploadDatasetArchive(file, { signal: controller.signal }).catch((reason) => reason);

        expect(isAbortError(error)).toBe(true);
        expect(consumed).toEqual([]);
        expect(deleted).toEqual([UPLOAD_ID]);
    });

    it('discards the upload when the server cannot consume it', async () => {
        const { deleted } = mockEndpoints();
        server.use(
            http.post('/api/staged_datasets:from-upload', () =>
                // @ts-expect-error The 422 error response is not part of the typed success schema
                HttpResponse.json({ detail: 'Unsupported archive' }, { status: 422 })
            )
        );

        await expect(uploadDatasetArchive(file)).rejects.toMatchObject({ detail: 'Unsupported archive' });
        expect(deleted).toEqual([UPLOAD_ID]);
    });

    it('surfaces the server error detail when the transfer fails', async () => {
        mockEndpoints();
        tus.onStart = (upload) => {
            const response = {
                getStatus: () => 413,
                getHeader: () => undefined,
                getBody: () => JSON.stringify({ detail: 'Upload exceeds the maximum size' }),
                getUnderlyingObject: () => undefined,
            };

            upload.options.onError?.(
                Object.assign(new DetailedError('tus: unexpected response'), { originalResponse: response })
            );
        };

        await expect(uploadDatasetArchive(file)).rejects.toThrow('Upload exceeds the maximum size');
    });
});
