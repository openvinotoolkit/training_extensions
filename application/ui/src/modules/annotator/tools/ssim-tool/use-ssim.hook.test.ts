// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act, waitFor } from '@testing-library/react';
import { wrap } from 'comlink';

import { renderHook } from '../../../../test-utils/render';
import { useSSIM } from './use-ssim.hook';

vi.mock('comlink', async (importOriginal) => ({
    ...(await importOriginal<typeof import('comlink')>()),
    wrap: vi.fn(),
}));

class FakeWorker {
    static instances: FakeWorker[] = [];
    terminate = vi.fn();

    constructor() {
        FakeWorker.instances.push(this);
    }
}

const template = { type: 'rectangle' as const, x: 10, y: 10, width: 20, height: 20 };

describe('useSSIM', () => {
    beforeEach(() => {
        FakeWorker.instances = [];
        vi.stubGlobal('Worker', FakeWorker);
    });

    afterEach(() => {
        vi.unstubAllGlobals();
    });

    it('replaces the worker and keeps the template when a run fails', async () => {
        const executeSSIM = vi.fn().mockRejectedValue(new Error('worker crashed'));
        vi.mocked(wrap).mockReturnValue({ build: () => Promise.resolve({ executeSSIM }) } as never);

        const { result } = renderHook(() => useSSIM());

        await waitFor(() => expect(result.current.worker).toBeDefined());

        act(() => {
            result.current.runSSIM({
                imageData: { width: 100, height: 100 } as ImageData,
                template,
                existingAnnotations: [],
                autoMergeDuplicates: true,
                shapeType: 'rectangle',
            });
        });

        await waitFor(() => expect(result.current.isError).toBe(true));

        expect(result.current.error?.message).toBe('worker crashed');
        expect(result.current.toolState.shapes).toEqual([template]);
        expect(FakeWorker.instances[0].terminate).toHaveBeenCalled();

        await waitFor(() => expect(FakeWorker.instances).toHaveLength(2));
        await waitFor(() => expect(result.current.isLoading).toBe(false));
        expect(FakeWorker.instances[1].terminate).not.toHaveBeenCalled();
    });

    it('rebuilds the worker on the next run when the first build failed', async () => {
        const executeSSIM = vi.fn().mockResolvedValue([]);
        vi.mocked(wrap)
            .mockReturnValueOnce({ build: () => Promise.reject(new Error('wasm failed')) } as never)
            .mockReturnValue({ build: () => Promise.resolve({ executeSSIM }) } as never);

        const { result } = renderHook(() => useSSIM());

        await waitFor(() => expect(result.current.error?.message).toBe('wasm failed'));

        act(() => {
            result.current.runSSIM({
                imageData: { width: 100, height: 100 } as ImageData,
                template,
                existingAnnotations: [],
                autoMergeDuplicates: true,
                shapeType: 'rectangle',
            });
        });

        await waitFor(() => expect(result.current.worker).toBeDefined());
        expect(FakeWorker.instances).toHaveLength(2);
        expect(result.current.toolState.shapes).toEqual([template]);
    });
});
