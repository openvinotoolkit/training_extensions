// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act, renderHook, waitFor } from '@testing-library/react';
import { getMockedShape } from 'mocks/mock-annotation';
import { getMockedMediaImage } from 'mocks/mock-media';

import type { InteractiveAnnotationPoint } from './segment-anything.interface';
import { useSegmentAnythingPrompt } from './use-segment-anything-prompt.hook';

const ROI = { x: 0, y: 0, width: 1000, height: 1000 };
const HIT_RADIUS = 5;

// The mask encodes how many points it was decoded from, to tell which prompt it belongs to.
const decodingQueryFn = rs.fn(async (points: InteractiveAnnotationPoint[], _previousMask?: Float32Array) => ({
    shapes: [getMockedShape()],
    lowResMask: new Float32Array([points.length]),
}));

const renderPrompt = () =>
    renderHook((props) => useSegmentAnythingPrompt(props), {
        initialProps: { decodingQueryFn, roi: ROI, mediaItem: getMockedMediaImage({ id: 'image-1' }) },
    });

const lastDecodingCall = () => decodingQueryFn.mock.lastCall;

describe('useSegmentAnythingPrompt', () => {
    beforeEach(() => {
        decodingQueryFn.mockClear();
    });

    it('previews the object under the cursor', async () => {
        const { result } = renderPrompt();

        act(() => result.current.hover({ x: 10, y: 20 }));

        await waitFor(() => expect(result.current.previewShapes).toEqual([getMockedShape()]));
        expect(lastDecodingCall()).toEqual([[{ x: 10, y: 20, positive: true }], undefined]);
    });

    it('stops following the cursor once points are placed', async () => {
        const { result } = renderPrompt();

        act(() => result.current.togglePoint({ x: 10, y: 20 }, true, HIT_RADIUS));
        await waitFor(() => expect(result.current.isDecoding).toBe(false));

        decodingQueryFn.mockClear();
        act(() => result.current.hover({ x: 500, y: 500 }));
        act(() => result.current.clearHover());

        expect(decodingQueryFn).not.toHaveBeenCalled();
        expect(result.current.previewShapes).toHaveLength(1);
    });

    it('refines the object with the previous mask when adding a point, and starts over when removing one', async () => {
        const { result } = renderPrompt();
        const positive = { x: 10, y: 20, positive: true };
        const negative = { x: 300, y: 300, positive: false };

        act(() => result.current.togglePoint(positive, true, HIT_RADIUS));
        await waitFor(() => expect(result.current.isDecoding).toBe(false));
        expect(lastDecodingCall()).toEqual([[positive], undefined]);

        act(() => result.current.togglePoint(negative, false, HIT_RADIUS));
        await waitFor(() => expect(result.current.isDecoding).toBe(false));
        expect(lastDecodingCall()).toEqual([[positive, negative], new Float32Array([1])]);

        act(() => result.current.togglePoint({ x: 302, y: 301 }, true, HIT_RADIUS));
        await waitFor(() => expect(result.current.isDecoding).toBe(false));
        expect(result.current.points).toEqual([positive]);
        expect(lastDecodingCall()).toEqual([[positive], undefined]);
    });

    it('clears the prompt when the last point is removed', async () => {
        const { result } = renderPrompt();

        act(() => result.current.togglePoint({ x: 10, y: 20 }, true, HIT_RADIUS));
        await waitFor(() => expect(result.current.previewShapes).toHaveLength(1));

        act(() => result.current.togglePoint({ x: 10, y: 20 }, true, HIT_RADIUS));

        expect(result.current.points).toEqual([]);
        expect(result.current.previewShapes).toEqual([]);
    });

    it('clears the prompt when the media item changes', async () => {
        const { result, rerender } = renderPrompt();

        act(() => result.current.togglePoint({ x: 10, y: 20 }, true, HIT_RADIUS));
        await waitFor(() => expect(result.current.previewShapes).toHaveLength(1));

        rerender({ decodingQueryFn, roi: ROI, mediaItem: getMockedMediaImage({ id: 'image-2' }) });

        expect(result.current.points).toEqual([]);
        expect(result.current.previewShapes).toEqual([]);
    });

    it('ignores a decode that resolves after the media item changed', async () => {
        let resolveDecode: (value: Awaited<ReturnType<typeof decodingQueryFn>>) => void = () => undefined;
        decodingQueryFn.mockImplementationOnce(() => new Promise((resolve) => (resolveDecode = resolve)));

        const { result, rerender } = renderPrompt();

        act(() => result.current.togglePoint({ x: 10, y: 20 }, true, HIT_RADIUS));
        await waitFor(() => expect(decodingQueryFn).toHaveBeenCalledTimes(1));
        expect(result.current.isDecoding).toBe(true);

        rerender({ decodingQueryFn, roi: ROI, mediaItem: getMockedMediaImage({ id: 'image-2' }) });

        await act(async () => resolveDecode({ shapes: [getMockedShape()], lowResMask: new Float32Array([1]) }));

        expect(result.current.points).toEqual([]);
        expect(result.current.previewShapes).toEqual([]);
        expect(result.current.isDecoding).toBe(false);
    });
});
