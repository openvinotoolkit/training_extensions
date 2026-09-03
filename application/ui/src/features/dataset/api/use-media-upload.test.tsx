// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act, waitFor } from '@testing-library/react';
import { getMockedMediaImage } from 'mocks/mock-media';
import { HttpResponse } from 'msw';
import { renderHook } from 'test-utils/render';
import { v4 as uuid } from 'uuid';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { MediaUploadProvider, useMediaUploadState } from '../providers/media-upload-provider.component';
import { computeSummary } from '../providers/media-upload-reducer';
import { MEDIA_UPLOAD_CONCURRENCY, useMediaUpload } from './use-media-upload';

const useMediaUploadProgress = () => {
    const upload = useMediaUpload();
    const state = useMediaUploadState();

    return { upload, state, uploadProgress: computeSummary(state.items, state.isUploading) };
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
    it('uploads all selected files', async () => {
        const uploadedFileNames: string[] = [];

        server.use(
            http.post('/api/projects/{project_id}/dataset/media', async ({ request, params }) => {
                const formData = await request.formData();
                const file = formData.get('file');

                uploadedFileNames.push((file as File).name);
                expect(params.project_id).toBe('123');

                return HttpResponse.json(getMockedMediaImage({ id: crypto.randomUUID() }), { status: 201 });
            })
        );

        const { result } = renderUpload();

        const files = [
            new File(['file-1'], 'image-1.jpg', { type: 'image/jpeg' }),
            new File(['file-2'], 'image-2.jpg', { type: 'image/jpeg' }),
        ];

        await uploadMediaAndWaitForCompletion(
            result.current.upload.uploadMedia,
            files,
            () => result.current.uploadProgress.isUploading
        );

        expect(uploadedFileNames).toEqual(['image-1.jpg', 'image-2.jpg']);
    });

    it('does not exceed configured upload concurrency', async () => {
        let runningUploads = 0;
        let maxRunningUploads = 0;

        server.use(
            http.post('/api/projects/{project_id}/dataset/media', async () => {
                runningUploads += 1;
                maxRunningUploads = Math.max(maxRunningUploads, runningUploads);

                await new Promise((resolve) => {
                    setTimeout(resolve, 20);
                });

                runningUploads -= 1;

                return HttpResponse.json(getMockedMediaImage({ id: uuid() }), { status: 201 });
            })
        );

        const { result } = renderUpload();

        const mockFiles = Array.from(
            { length: 12 },
            (_, index) => new File([`file-${index}`], `image-${index}.jpg`, { type: 'image/jpeg' })
        );

        await uploadMediaAndWaitForCompletion(
            result.current.upload.uploadMedia,
            mockFiles,
            () => result.current.uploadProgress.isUploading
        );

        expect(maxRunningUploads).toBeLessThanOrEqual(MEDIA_UPLOAD_CONCURRENCY);
        expect(result.current.uploadProgress.completed).toBe(12);
    });

    it('tracks upload progress counters', async () => {
        let requestCount = 0;

        server.use(
            http.post('/api/projects/{project_id}/dataset/media', async () => {
                requestCount += 1;

                if (requestCount === 2) {
                    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                    // @ts-expect-error
                    return HttpResponse.json({ detail: 'Upload failed' }, { status: 400 });
                }

                return HttpResponse.json(getMockedMediaImage({ id: uuid() }), { status: 201 });
            })
        );

        const { result } = renderUpload();

        const files = [
            new File(['ok-file'], 'ok.jpg', { type: 'image/jpeg' }),
            new File(['broken-file'], 'broken.jpg', { type: 'image/jpeg' }),
        ];

        await uploadMediaAndWaitForCompletion(
            result.current.upload.uploadMedia,
            files,
            () => result.current.uploadProgress.isUploading
        );

        expect(result.current.uploadProgress).toEqual({
            total: 2,
            completed: 2,
            succeeded: 1,
            failed: 1,
            isUploading: false,
        });
    });

    it('tracks per-file status and error messages', async () => {
        server.use(
            http.post('/api/projects/{project_id}/dataset/media', async ({ request }) => {
                const formData = await request.formData();
                const file = formData.get('file') as File;

                if (file.name === 'broken.jpg') {
                    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                    // @ts-expect-error
                    return HttpResponse.json({ detail: 'Upload failed' }, { status: 400 });
                }

                return HttpResponse.json(getMockedMediaImage({ id: uuid() }), { status: 201 });
            })
        );

        const { result } = renderUpload();

        const files = [
            new File(['ok-file'], 'ok.jpg', { type: 'image/jpeg' }),
            new File(['broken-file'], 'broken.jpg', { type: 'image/jpeg' }),
        ];

        await uploadMediaAndWaitForCompletion(
            result.current.upload.uploadMedia,
            files,
            () => result.current.uploadProgress.isUploading
        );

        const items = result.current.state.items;
        expect(items).toHaveLength(2);
        expect(items[0]).toMatchObject({ name: 'ok.jpg', status: 'uploaded' });
        expect(items[1]).toMatchObject({ name: 'broken.jpg', status: 'failed' });
        expect(items[1].errorMessage).toBeTruthy();
    });
});
