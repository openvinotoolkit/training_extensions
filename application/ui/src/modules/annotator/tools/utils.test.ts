// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { canRasteriseAtFullSize, getImageData } from './utils';

const createImage = (width: number, height: number, naturalWidth = width, naturalHeight = height): HTMLImageElement =>
    ({ width, height, naturalWidth, naturalHeight }) as HTMLImageElement;

const createCanvasContext = () => {
    return {
        filter: '',
        drawImage: rs.fn(),
        getImageData: rs.fn((_x: number, _y: number, width: number, height: number) => {
            return {
                width,
                height,
                data: new Uint8ClampedArray(width * height * 4),
                colorSpace: 'srgb',
            } as ImageData;
        }),
    } as unknown as CanvasRenderingContext2D;
};

const mockCanvasCreation = (contexts: Array<CanvasRenderingContext2D | null>) => {
    const originalCreateElement = document.createElement.bind(document);

    return rs.spyOn(document, 'createElement').mockImplementation(((tagName: string) => {
        if (tagName !== 'canvas') {
            return originalCreateElement(tagName as keyof HTMLElementTagNameMap);
        }

        const context = contexts.shift() ?? null;
        return {
            width: 0,
            height: 0,
            getContext: rs.fn(() => context),
        } as unknown as HTMLCanvasElement;
    }) as typeof document.createElement);
};

afterEach(() => {
    rs.restoreAllMocks();
});

describe('canRasteriseAtFullSize', () => {
    it('accepts the exact boundary size 16384x16384', () => {
        expect(canRasteriseAtFullSize(16384, 16384)).toBe(true);
    });

    it('rejects dimensions one pixel over the boundary', () => {
        expect(canRasteriseAtFullSize(16385, 16384)).toBe(false);
        expect(canRasteriseAtFullSize(16384, 16385)).toBe(false);
    });
});

describe('getImageData', () => {
    it('returns 1x1 placeholder for empty images', () => {
        const imageData = getImageData(createImage(0, 0));

        expect(imageData.width).toBe(1);
        expect(imageData.height).toBe(1);
    });

    it('returns a 1x1 placeholder for oversized images', () => {
        const imageData = getImageData(createImage(19156, 15010));

        expect(imageData.width).toBe(1);
        expect(imageData.height).toBe(1);
    });

    it('rasterises images within the canvas limit at full size', () => {
        const context = createCanvasContext();
        mockCanvasCreation([context]);

        const imageData = getImageData(createImage(4000, 3000));

        expect(imageData.width).toBe(4000);
        expect(imageData.height).toBe(3000);
    });

    it('returns a 1x1 placeholder when the canvas context is unavailable', () => {
        mockCanvasCreation([null]);

        const imageData = getImageData(createImage(4000, 3000));

        expect(imageData.width).toBe(1);
        expect(imageData.height).toBe(1);
    });
});
