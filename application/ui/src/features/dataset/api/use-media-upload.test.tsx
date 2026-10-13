// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { uploadDatasetMedia } from '@/api';
import { act, waitFor } from '@testing-library/react';
import { getMockedMediaImage } from 'mocks/mock-media';
import { renderHook } from 'test-utils/render';
import { beforeEach, vi } from 'vitest';

import { useUploadActions } from '../hooks/use-upload-actions';
import { useMediaUploadState } from '../providers/media-upload-context';
import { MediaUploadProvider } from '../providers/media-upload-provider.component';
import { computeSummary } from '../providers/media-upload-reducer';
import { MEDIA_UPLOAD_CONCURRENCY, useMediaUpload } from './use-media-upload';

vi.mock('@/api', async (importOriginal) => ({
    ...(await importOriginal<typeof import('@/api')>()),
    uploadDatasetMedia: vi.fn(),
}));
const uploadMock = vi.mocked(uploadDatasetMedia);

const useMediaUploadProgress = () => {
    const upload = useMediaUpload();
    const actions = useUploadActions();
    const state = useMediaUploadState();

    return { upload, actions, state, uploadProgress: computeSummary(state.items) };
};

const renderUpload = () => renderHook(() => useMediaUploadProgress(), { wrapper: MediaUploadProvider });
const uploadMediaAndWaitForCompletion = async (
    uploadMedia: (files: File[]) => Promise<unknown>,
    files: File[],
    isUploading: () => boolean
) => {
    await act(async () => {
        await uploadMedia(files);
    });

    await waitFor(() => {
        expect(isUploading()).toBe(false);
    });
};

describe('useMediaUpload', () => {
    beforeEach(() => {
        uploadMock.mockReset();
    });

    it('tracks transferred bytes separately from server processing', async () => {
        let finishProcessing: ((media: ReturnType<typeof getMockedMediaImage>) => void) | undefined;
        uploadMock.mockImplementation(
            async (_projectId, file, options) =>
                new Promise((resolve) => {
                    options?.onProgress?.(2);
                    options?.onProgress?.(file.size);
                    finishProcessing = resolve;
                })
        );
        const { result } = renderUpload();
        const file = new File(['file'], 'image.jpg');
        let pending: Promise<unknown>;
        act(() => {
            pending = result.current.upload.uploadMedia([file]);
        });

        await waitFor(() =>
            expect(result.current.state.items[0]).toMatchObject({
                bytesSent: file.size,
                status: 'processing',
            })
        );
        await act(async () => {
            finishProcessing?.(getMockedMediaImage({ id: crypto.randomUUID() }));
            await pending;
        });
        expect(result.current.state.items[0].status).toBe('uploaded');
    });

    it('uploads all selected files', async () => {
        const uploadedFileNames: string[] = [];

        uploadMock.mockImplementation(async (projectId, file) => {
            uploadedFileNames.push(file.name);
            expect(projectId).toBe('123');
            return getMockedMediaImage({ id: crypto.randomUUID() });
        });

        const { result } = renderUpload();

        const files = [
            new File(['file-1'], 'image-1.jpg', { type: 'image/jpeg' }),
            new File(['file-2'], 'image-2.jpg', { type: 'image/jpeg' }),
        ];

        await uploadMediaAndWaitForCompletion(
            result.current.upload.uploadMedia,
            files,
            () => result.current.state.isUploading
        );

        expect(uploadedFileNames).toEqual(['image-1.jpg', 'image-2.jpg']);
    });

    it('does not exceed configured upload concurrency', async () => {
        let runningUploads = 0;
        let maxRunningUploads = 0;

        uploadMock.mockImplementation(async () => {
            runningUploads += 1;
            maxRunningUploads = Math.max(maxRunningUploads, runningUploads);
            await new Promise((resolve) => setTimeout(resolve, 20));
            runningUploads -= 1;
            return getMockedMediaImage({ id: crypto.randomUUID() });
        });

        const { result } = renderUpload();

        const mockFiles = Array.from(
            { length: 12 },
            (_, index) => new File([`file-${index}`], `image-${index}.jpg`, { type: 'image/jpeg' })
        );

        await uploadMediaAndWaitForCompletion(
            result.current.upload.uploadMedia,
            mockFiles,
            () => result.current.state.isUploading
        );

        expect(maxRunningUploads).toBeLessThanOrEqual(MEDIA_UPLOAD_CONCURRENCY);
        expect(result.current.uploadProgress.succeeded).toBe(12);
    });

    it('tracks upload progress counters', async () => {
        let requestCount = 0;

        uploadMock.mockImplementation(async () => {
            requestCount += 1;
            if (requestCount === 2) {
                throw new Error('Upload failed');
            }
            return getMockedMediaImage({ id: crypto.randomUUID() });
        });

        const { result } = renderUpload();

        const files = [
            new File(['ok-file'], 'ok.jpg', { type: 'image/jpeg' }),
            new File(['broken-file'], 'broken.jpg', { type: 'image/jpeg' }),
        ];

        await uploadMediaAndWaitForCompletion(
            result.current.upload.uploadMedia,
            files,
            () => result.current.state.isUploading
        );

        expect(result.current.uploadProgress).toEqual({
            total: 2,
            succeeded: 1,
            failed: 1,
            cancelled: 0,
        });
    });

    it('tracks per-file status and error messages', async () => {
        uploadMock.mockImplementation(async (_projectId, file) => {
            if (file.name === 'broken.jpg') {
                throw new Error('Upload failed');
            }
            return getMockedMediaImage({ id: crypto.randomUUID() });
        });

        const { result } = renderUpload();

        const files = [
            new File(['ok-file'], 'ok.jpg', { type: 'image/jpeg' }),
            new File(['broken-file'], 'broken.jpg', { type: 'image/jpeg' }),
        ];

        await uploadMediaAndWaitForCompletion(
            result.current.upload.uploadMedia,
            files,
            () => result.current.state.isUploading
        );

        const items = result.current.state.items;
        expect(items).toHaveLength(2);
        expect(items[0]).toMatchObject({ name: 'ok.jpg', status: 'uploaded' });
        expect(items[1]).toMatchObject({ name: 'broken.jpg', status: 'failed' });
        expect(items[1].errorMessage).toBeTruthy();
    });
    it('cancels individual uploads, including queued ones, without marking them as failed', async () => {
        const signals = new Map<string, AbortSignal | undefined>();
        uploadMock.mockImplementation(
            async (_projectId, file, options) =>
                new Promise((resolve, reject) => {
                    signals.set(file.name, options?.signal);
                    options?.onProgress?.(1);

                    if (file.name === 'keep.jpg') {
                        resolve(getMockedMediaImage({ id: crypto.randomUUID() }));
                        return;
                    }

                    options?.signal?.addEventListener('abort', () =>
                        reject(new DOMException('The upload was cancelled.', 'AbortError'))
                    );
                })
        );
        const { result } = renderUpload();
        const files = Array.from(
            { length: MEDIA_UPLOAD_CONCURRENCY },
            (_, index) => new File(['file'], index === 0 ? 'keep.jpg' : `cancel-${index}.jpg`)
        );
        const queuedFile = new File(['file'], 'queued.jpg');
        let pending: Promise<unknown>;

        act(() => {
            pending = result.current.upload.uploadMedia([...files, queuedFile]);
        });
        await waitFor(() => expect(signals.size).toBe(MEDIA_UPLOAD_CONCURRENCY));

        await act(async () => {
            const cancellableIds = result.current.state.items
                .filter((item) => item.name !== 'keep.jpg')
                .map((item) => item.id);

            result.current.actions.cancelItems(cancellableIds);
            await pending;
        });

        expect(signals.get('cancel-1.jpg')?.aborted).toBe(true);
        expect(signals.has('queued.jpg')).toBe(false);
        expect(result.current.uploadProgress).toEqual({
            total: MEDIA_UPLOAD_CONCURRENCY + 1,
            succeeded: 1,
            failed: 0,
            cancelled: MEDIA_UPLOAD_CONCURRENCY,
        });
        expect(result.current.state.isUploading).toBe(false);
    });

    it('does not cancel items whose transfer already completed', async () => {
        let finishProcessing: ((media: ReturnType<typeof getMockedMediaImage>) => void) | undefined;
        let signal: AbortSignal | undefined;
        uploadMock.mockImplementation(
            async (_projectId, file, options) =>
                new Promise((resolve) => {
                    signal = options?.signal;
                    options?.onProgress?.(file.size);
                    finishProcessing = resolve;
                })
        );
        const { result } = renderUpload();
        let pending: Promise<unknown>;

        act(() => {
            pending = result.current.upload.uploadMedia([new File(['file'], 'image.jpg')]);
        });
        await waitFor(() => expect(result.current.state.items[0].status).toBe('processing'));

        act(() => result.current.actions.cancelItems([result.current.state.items[0].id]));
        expect(result.current.state.items[0].status).toBe('processing');
        expect(signal?.aborted).toBe(false);

        await act(async () => {
            finishProcessing?.(getMockedMediaImage({ id: crypto.randomUUID() }));
            await pending;
        });
        expect(result.current.state.items[0].status).toBe('uploaded');
    });
});
