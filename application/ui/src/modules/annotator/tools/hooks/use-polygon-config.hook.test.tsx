// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { renderHook } from '@testing-library/react';

import type { Polygon } from '../../../../shared/types';
import { usePolygonConfig } from './use-polygon-config.hook';

const mockOptimizeSegments = vi.fn();
let mockWorker:
    { optimizeSegments: typeof mockOptimizeSegments; loadImage: () => void; cleanImg: () => void } | undefined;

vi.mock('./use-intelligent-scissors-worker.hook', () => ({
    useIntelligentScissorsWorker: () => ({ worker: mockWorker }),
}));

vi.mock('../polygon-tool/polygon-state-provider.component', () => ({
    usePolygonState: () => ({
        segments: [[{ x: 0, y: 0 }]],
        setSegments: vi.fn(),
        pointerLine: [],
        setPointerLine: vi.fn(),
        lassoSegment: [],
        setLassoSegment: vi.fn(),
        undoRedoActions: { reset: vi.fn() },
    }),
}));

const drawnPolygon: Polygon = {
    type: 'polygon',
    points: [
        { x: 0, y: 0 },
        { x: 10, y: 0 },
        { x: 10, y: 10 },
    ],
};

const renderPolygonConfig = () =>
    renderHook(() =>
        usePolygonConfig({
            zoom: 1,
            image: { width: 1, height: 1, data: new Uint8ClampedArray(4) } as ImageData,
            canvasRef: { current: document.createElementNS('http://www.w3.org/2000/svg', 'rect') },
        })
    );

describe('usePolygonConfig', () => {
    beforeEach(() => {
        mockOptimizeSegments.mockReset();
        mockWorker = { optimizeSegments: mockOptimizeSegments, loadImage: vi.fn(), cleanImg: vi.fn() };
    });

    it('returns the optimized polygon from the worker', async () => {
        const optimizedPoints = [
            { x: 0, y: 0 },
            { x: 10, y: 10 },
            { x: 0, y: 10 },
        ];
        mockOptimizeSegments.mockResolvedValue({ shapeType: 'polygon', points: optimizedPoints });

        const { result } = renderPolygonConfig();

        await expect(result.current.optimizePolygonOrSegments(drawnPolygon)).resolves.toEqual({
            type: 'polygon',
            points: optimizedPoints,
        });
        expect(mockOptimizeSegments).toHaveBeenCalledWith([[{ x: 0, y: 0 }], drawnPolygon.points.slice(1)]);
    });

    it('keeps the drawn polygon when the worker is not ready yet', async () => {
        mockWorker = undefined;

        const { result } = renderPolygonConfig();

        await expect(result.current.optimizePolygonOrSegments(drawnPolygon)).resolves.toEqual(drawnPolygon);
    });

    it('keeps the drawn polygon when the worker fails to optimize it', async () => {
        mockOptimizeSegments.mockRejectedValue(new Error('opencv failure'));

        const { result } = renderPolygonConfig();

        await expect(result.current.optimizePolygonOrSegments(drawnPolygon)).resolves.toEqual(drawnPolygon);
    });
});
