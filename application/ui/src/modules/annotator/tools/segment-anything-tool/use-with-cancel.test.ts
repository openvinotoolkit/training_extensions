// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act } from '@testing-library/react';
import { getMockedShape } from 'mocks/mock-annotation';
import { renderHook } from 'test-utils/render';

import { Shape } from '../../../../shared/types';
import { InteractiveAnnotationPoint } from './segment-anything.interface';
import { useWithCancel } from './use-with-cancel';

const getMockedInteractivePoint = (x: number, y: number, positive = true): InteractiveAnnotationPoint => ({
    x,
    y,
    positive,
});

const makeControllableFn = () => {
    const resolvers: Array<(shapes: Shape[]) => void> = [];

    const fn = rs.fn(
        (_points: InteractiveAnnotationPoint[]) =>
            new Promise<Shape[]>((res) => {
                resolvers.push(res);
            })
    );

    return {
        fn,
        resolveCall: (index: number, shapes: Shape[] = [getMockedShape()]) => resolvers[index]?.(shapes),
    };
};

describe('useWithCancel', () => {
    it('returns the resolved value when fn resolves normally', async () => {
        const shapes = [getMockedShape()];
        const fn = rs.fn(async (_points: InteractiveAnnotationPoint[]) => shapes);

        const { result } = renderHook(() => useWithCancel(fn));

        let returnedShapes: Shape[] | null = null;
        await act(async () => {
            returnedShapes = await result.current.call([getMockedInteractivePoint(1, 2)]);
        });

        expect(returnedShapes).toBe(shapes);
    });

    it('passes the correct arguments to fn', async () => {
        const points = [getMockedInteractivePoint(3, 4, false), getMockedInteractivePoint(7, 8)];
        const previousMask = new Float32Array([0.5]);
        const fn = rs.fn().mockResolvedValue([]);

        const { result } = renderHook(() => useWithCancel(fn));

        await act(async () => {
            await result.current.call(points, previousMask);
        });

        expect(fn).toHaveBeenCalledWith(points, previousMask);
    });

    it('rejects with AbortError when cancel() is called while fn is in-flight', async () => {
        const controlled = makeControllableFn();
        const { result } = renderHook(() => useWithCancel(controlled.fn));

        let callPromise: Promise<Shape[]> | null = null;

        act(() => {
            callPromise = result.current.call([getMockedInteractivePoint(1, 2)]);
        });

        // Cancel while fn is still pending.
        act(() => {
            result.current.cancel();
        });

        // Now let fn finish — the AbortError should be thrown despite fn resolving.
        act(() => {
            controlled.resolveCall(0);
        });

        await expect(callPromise).rejects.toMatchObject({ name: 'AbortError', message: 'Request aborted' });
    });

    it('rejects the first call with AbortError when a second call is made', async () => {
        const controlled = makeControllableFn();
        const { result } = renderHook(() => useWithCancel(controlled.fn));

        let firstCallPromise: Promise<Shape[]> | null = null;
        let secondCallPromise: Promise<Shape[]> | null = null;

        act(() => {
            firstCallPromise = result.current.call([getMockedInteractivePoint(0, 0)]);
        });

        // Trigger a second call — this should abort the first.
        act(() => {
            secondCallPromise = result.current.call([getMockedInteractivePoint(1, 1)]);
        });

        const secondCallShapes = [getMockedShape()];

        // Resolve both fn invocations so both call promises can settle.
        act(() => {
            controlled.resolveCall(0);
            controlled.resolveCall(1, secondCallShapes);
        });

        await expect(firstCallPromise).rejects.toMatchObject({ name: 'AbortError', message: 'Request aborted' });

        // The second call should resolve normally.
        const secondResult = await secondCallPromise;
        expect(secondResult).toEqual(secondCallShapes);
    });

    it('does not throw when cancel() is called before any call', () => {
        const fn = rs.fn().mockResolvedValue([]);
        const { result } = renderHook(() => useWithCancel(fn));

        expect(() => {
            result.current.cancel();
        }).not.toThrow();
    });

    it('rejects an in-flight call with AbortError when the component unmounts', async () => {
        const controlled = makeControllableFn();
        const { result, unmount } = renderHook(() => useWithCancel(controlled.fn));

        let callPromise: Promise<Shape[]> | null = null;
        act(() => {
            callPromise = result.current.call([getMockedInteractivePoint(5, 5)]);
        });

        // Unmount triggers the useEffect cleanup which calls cancel().
        act(() => {
            unmount();
        });

        // Resolve fn after unmount — the AbortError should still be thrown.
        act(() => {
            controlled.resolveCall(0);
        });

        await expect(callPromise).rejects.toMatchObject({ name: 'AbortError', message: 'Request aborted' });
    });

    it('works normally after a previous cancel()', async () => {
        const shapes = [getMockedShape()];
        const fn = rs.fn(async (_points: InteractiveAnnotationPoint[]) => shapes);

        const { result } = renderHook(() => useWithCancel(fn));

        act(() => {
            result.current.cancel();
        });

        let returnedShapes: Shape[] | null = null;
        await act(async () => {
            returnedShapes = await result.current.call([getMockedInteractivePoint(2, 3)]);
        });

        expect(returnedShapes).toBe(shapes);
    });
});
