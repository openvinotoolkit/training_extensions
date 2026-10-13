// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { uploadSourceVideo } from '@/api';
import { act, renderHook, waitFor } from '@testing-library/react';

import { useVideoFileUpload } from './use-video-file-upload.hook';

vi.mock('@/api', async (importOriginal) => ({
    ...(await importOriginal<typeof import('@/api')>()),
    uploadSourceVideo: vi.fn(),
}));

const buildFormData = (): FormData => {
    const formData = new FormData();
    formData.append('video_path', '');
    formData.append('video_file', new File(['fake-video-bytes'], 'sample.mp4', { type: 'video/mp4' }));

    return formData;
};

describe('useVideoFileUpload', () => {
    it('exposes the transfer progress while uploading and resets it afterwards', async () => {
        let finish: (() => void) | undefined;
        vi.mocked(uploadSourceVideo).mockImplementation(
            async (_file, options) =>
                new Promise((resolve) => {
                    options?.onProgress?.(4);
                    finish = () => resolve({ video_path: '/data/sample.mp4' });
                })
        );
        const { result } = renderHook(() => useVideoFileUpload());
        const formData = buildFormData();
        let pending: ReturnType<typeof result.current.prepareFormData> | undefined;

        act(() => {
            pending = result.current.prepareFormData(formData);
        });
        await waitFor(() => expect(result.current.progress).toEqual({ bytesSent: 4, bytesTotal: 16 }));

        await act(async () => {
            finish?.();
            await pending;
        });

        expect(result.current.progress).toBeNull();
        expect(formData.get('video_path')).toBe('/data/sample.mp4');
    });

    it('aborts the transfer when cancelled', async () => {
        vi.mocked(uploadSourceVideo).mockImplementation(
            async (_file, options) =>
                new Promise((_resolve, reject) => {
                    options?.onProgress?.(1);
                    options?.signal?.addEventListener('abort', () =>
                        reject(new DOMException('The upload was cancelled.', 'AbortError'))
                    );
                })
        );
        const { result } = renderHook(() => useVideoFileUpload());
        let pending: ReturnType<typeof result.current.prepareFormData> | undefined;

        act(() => {
            pending = result.current.prepareFormData(buildFormData());
        });
        await waitFor(() => expect(result.current.progress).not.toBeNull());

        await act(async () => {
            result.current.cancel();
            await expect(pending).rejects.toMatchObject({ name: 'AbortError' });
        });

        expect(result.current.progress).toBeNull();
    });

    it('aborts the transfer when unmounted', async () => {
        let signal: AbortSignal | undefined;
        vi.mocked(uploadSourceVideo).mockImplementation(
            async (_file, options) =>
                new Promise(() => {
                    signal = options?.signal;
                })
        );
        const { result, unmount } = renderHook(() => useVideoFileUpload());

        act(() => {
            void result.current.prepareFormData(buildFormData());
        });
        await waitFor(() => expect(signal).toBeDefined());

        unmount();

        expect(signal?.aborted).toBe(true);
    });
});
