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
const imageData = { width: 100, height: 100 } as ImageData;
const runProps = { imageData, template, existingAnnotations: [], autoMergeDuplicates: true };

const toolMatch = (x: number, y: number) => ({
    shape: { x, y, width: 20, height: 20, shapeType: 'rect' as const },
    confidence: 0.95,
});

const mockWorker = (executeSSIM: ReturnType<typeof vi.fn>) => {
    vi.mocked(wrap).mockReturnValue({ build: () => Promise.resolve({ executeSSIM }) } as never);
};

const flushPromises = () => act(() => new Promise((resolve) => setTimeout(resolve, 0)));

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
        mockWorker(executeSSIM);

        const { result } = renderHook(() => useSSIM('rectangle'));

        await waitFor(() => expect(result.current.worker).toBeDefined());

        act(() => {
            result.current.runSSIM(runProps);
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

        const { result } = renderHook(() => useSSIM('rectangle'));

        await waitFor(() => expect(result.current.error?.message).toBe('wasm failed'));

        act(() => {
            result.current.runSSIM(runProps);
        });

        await waitFor(() => expect(result.current.worker).toBeDefined());
        expect(FakeWorker.instances).toHaveLength(2);
        expect(result.current.toolState.shapes).toEqual([template]);
    });

    it('converts the matches to polygons when the shape type is polygon', async () => {
        mockWorker(vi.fn().mockResolvedValue([toolMatch(60, 60)]));

        const { result } = renderHook(() => useSSIM('polygon'));

        await waitFor(() => expect(result.current.worker).toBeDefined());

        act(() => {
            result.current.runSSIM(runProps);
        });

        await waitFor(() => expect(result.current.toolState.shapes).toHaveLength(2));
        expect(result.current.toolState.shapes.every((shape) => shape.type === 'polygon')).toBe(true);
    });

    it('ignores the result of a run that was reset while in flight', async () => {
        const { promise, resolve } = Promise.withResolvers<ReturnType<typeof toolMatch>[]>();
        mockWorker(vi.fn().mockReturnValue(promise));

        const { result } = renderHook(() => useSSIM('rectangle'));

        await waitFor(() => expect(result.current.worker).toBeDefined());

        act(() => {
            result.current.runSSIM(runProps);
        });

        await waitFor(() => expect(result.current.isProcessing).toBe(true));

        act(() => {
            result.current.reset();
        });

        resolve([toolMatch(60, 60)]);
        await flushPromises();

        expect(result.current.toolState.shapes).toEqual([]);
        expect(result.current.isProcessing).toBe(false);
    });

    it('ignores the result of a run that was superseded by a newer one', async () => {
        const { promise: firstRun, resolve: resolveFirstRun } = Promise.withResolvers<ReturnType<typeof toolMatch>[]>();
        const executeSSIM = vi
            .fn()
            .mockReturnValueOnce(firstRun)
            .mockResolvedValueOnce([toolMatch(60, 60)]);
        mockWorker(executeSSIM);

        const { result } = renderHook(() => useSSIM('rectangle'));

        await waitFor(() => expect(result.current.worker).toBeDefined());

        act(() => {
            result.current.runSSIM(runProps);
        });
        act(() => {
            result.current.runSSIM(runProps);
        });

        await waitFor(() => expect(result.current.toolState.shapes).toHaveLength(2));

        resolveFirstRun([]);
        await flushPromises();

        expect(result.current.toolState.shapes).toHaveLength(2);
    });

    it('does not replace the worker when a superseded run rejects', async () => {
        const { promise: firstRun, reject: rejectFirstRun } = Promise.withResolvers<ReturnType<typeof toolMatch>[]>();
        const { promise: secondRun, resolve: resolveSecondRun } =
            Promise.withResolvers<ReturnType<typeof toolMatch>[]>();
        mockWorker(vi.fn().mockReturnValueOnce(firstRun).mockReturnValueOnce(secondRun));

        const { result } = renderHook(() => useSSIM('rectangle'));

        await waitFor(() => expect(result.current.worker).toBeDefined());

        act(() => {
            result.current.runSSIM(runProps);
        });
        act(() => {
            result.current.runSSIM(runProps);
        });

        rejectFirstRun(new Error('stale failure'));
        await flushPromises();

        expect(FakeWorker.instances).toHaveLength(1);
        expect(FakeWorker.instances[0].terminate).not.toHaveBeenCalled();

        resolveSecondRun([toolMatch(60, 60)]);

        await waitFor(() => expect(result.current.toolState.shapes).toHaveLength(2));
        expect(result.current.isError).toBe(false);
    });

    it('does not replace the worker when a run rejects after a reset', async () => {
        const { promise, reject } = Promise.withResolvers<ReturnType<typeof toolMatch>[]>();
        mockWorker(vi.fn().mockReturnValue(promise));

        const { result } = renderHook(() => useSSIM('rectangle'));

        await waitFor(() => expect(result.current.worker).toBeDefined());

        act(() => {
            result.current.runSSIM(runProps);
        });
        act(() => {
            result.current.reset();
        });

        reject(new Error('stale failure'));
        await flushPromises();

        expect(FakeWorker.instances).toHaveLength(1);
        expect(FakeWorker.instances[0].terminate).not.toHaveBeenCalled();
        expect(result.current.toolState.shapes).toEqual([]);
    });

    it('reruns the last template with the given overrides', async () => {
        const executeSSIM = vi.fn().mockResolvedValue([toolMatch(60, 60)]);
        mockWorker(executeSSIM);

        const { result } = renderHook(() => useSSIM('rectangle'));

        await waitFor(() => expect(result.current.worker).toBeDefined());

        act(() => {
            result.current.rerun({ autoMergeDuplicates: false });
        });

        expect(executeSSIM).not.toHaveBeenCalled();

        act(() => {
            result.current.runSSIM(runProps);
        });

        await waitFor(() => expect(result.current.toolState.shapes).toHaveLength(2));

        act(() => {
            result.current.rerun({ autoMergeDuplicates: false });
        });

        await waitFor(() => expect(executeSSIM).toHaveBeenCalledTimes(2));
        expect(executeSSIM.mock.lastCall?.[0]).toMatchObject({
            autoMergeDuplicates: false,
            template: { x: 10, y: 10, width: 20, height: 20 },
        });
    });

    it('limits the previewed shapes to the selected number of items', async () => {
        mockWorker(vi.fn().mockResolvedValue([toolMatch(40, 40), toolMatch(70, 70)]));

        const { result } = renderHook(() => useSSIM('rectangle'));

        await waitFor(() => expect(result.current.worker).toBeDefined());

        act(() => {
            result.current.runSSIM(runProps);
        });

        await waitFor(() => expect(result.current.toolState.shapes).toHaveLength(3));

        act(() => {
            result.current.updateToolState({ threshold: 1 });
        });

        expect(result.current.toolState.shapes).toEqual([template]);
        expect(result.current.toolState.matches).toHaveLength(3);
    });

    it('keeps the selected number of items and the preview while re-running', async () => {
        const { promise: rerunResult, resolve: resolveRerun } = Promise.withResolvers<ReturnType<typeof toolMatch>[]>();
        const executeSSIM = vi
            .fn()
            .mockResolvedValueOnce([toolMatch(40, 40), toolMatch(70, 70)])
            .mockReturnValueOnce(rerunResult);
        mockWorker(executeSSIM);

        const { result } = renderHook(() => useSSIM('rectangle'));

        await waitFor(() => expect(result.current.worker).toBeDefined());

        act(() => {
            result.current.runSSIM(runProps);
        });

        await waitFor(() => expect(result.current.toolState.shapes).toHaveLength(3));

        act(() => {
            result.current.updateToolState({ threshold: 2 });
        });
        act(() => {
            result.current.rerun({ autoMergeDuplicates: false });
        });

        await waitFor(() => expect(result.current.isProcessing).toBe(true));
        expect(result.current.toolState.shapes).toHaveLength(2);

        resolveRerun([toolMatch(40, 40), toolMatch(70, 70), toolMatch(10, 70)]);

        await waitFor(() => expect(result.current.toolState.matches).toHaveLength(4));
        expect(result.current.toolState.threshold).toBe(2);
        expect(result.current.toolState.shapes).toHaveLength(2);
    });

    it('clears the previous preview when running a new template', async () => {
        const { promise: secondRun } = Promise.withResolvers<ReturnType<typeof toolMatch>[]>();
        const executeSSIM = vi
            .fn()
            .mockResolvedValueOnce([toolMatch(60, 60)])
            .mockReturnValueOnce(secondRun);
        mockWorker(executeSSIM);

        const { result } = renderHook(() => useSSIM('rectangle'));

        await waitFor(() => expect(result.current.worker).toBeDefined());

        act(() => {
            result.current.runSSIM(runProps);
        });

        await waitFor(() => expect(result.current.toolState.shapes).toHaveLength(2));

        act(() => {
            result.current.runSSIM(runProps);
        });

        expect(result.current.toolState.shapes).toEqual([]);
    });
});
