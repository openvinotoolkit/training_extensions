// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { drawImageDataOnCanvas } from './draw-image-data-on-canvas';

const createCanvasContext = () => {
    return {
        putImageData: rs.fn(),
    } as unknown as CanvasRenderingContext2D;
};

describe('drawImageDataOnCanvas', () => {
    it('skips drawing when the input pixel buffer is invalid', () => {
        const ctx = createCanvasContext();
        const invalidImage = {
            width: 4,
            height: 4,
            data: new Uint8ClampedArray(10),
            colorSpace: 'srgb',
        } as ImageData;

        const didDraw = drawImageDataOnCanvas(ctx, invalidImage);

        expect(didDraw).toBe(false);
        expect(ctx.putImageData).not.toHaveBeenCalled();
    });

    it('draws valid image-shaped data directly', () => {
        const ctx = createCanvasContext();
        const image = {
            width: 2,
            height: 3,
            data: new Uint8ClampedArray(2 * 3 * 4).fill(7),
            colorSpace: 'srgb',
        } as ImageData;

        const didDraw = drawImageDataOnCanvas(ctx, image);

        expect(didDraw).toBe(true);
        expect(ctx.putImageData).toHaveBeenCalledWith(image, 0, 0);
    });
});
