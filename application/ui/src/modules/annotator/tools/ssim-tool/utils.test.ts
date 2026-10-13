// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Polygon, Rect, RegionOfInterest } from '../../../../shared/types';
import {
    convertRectToShape,
    convertToolMatchesToGetiMatches,
    DEFAULT_CONFIDENCE_THRESHOLD,
    filterSSIMResults,
    getExistingRects,
    guessNumberOfItemsThreshold,
    MAX_NUMBER_ITEMS,
    toToolRunSSIMProps,
    type SSIMMatch,
} from './utils';

const imageBounds: RegionOfInterest = { x: 0, y: 0, width: 1000, height: 1000 };

const rect = (x: number, y: number, width = 10, height = 10): Rect => ({ type: 'rectangle', x, y, width, height });

const match = (shape: Rect, confidence: number): SSIMMatch => ({ shape, confidence });

describe('SSIM utils', () => {
    describe('getExistingRects', () => {
        it('converts polygons to their bounding rect and skips full-image annotations', () => {
            const polygon: Polygon = {
                type: 'polygon',
                points: [
                    { x: 10, y: 20 },
                    { x: 30, y: 25 },
                    { x: 15, y: 50 },
                ],
            };

            expect(getExistingRects([rect(1, 2), polygon, { type: 'full_image' }])).toEqual([
                rect(1, 2),
                rect(10, 20, 20, 30),
            ]);
        });
    });

    describe('toToolRunSSIMProps', () => {
        it('converts the template and existing annotations to smart-tools rects', () => {
            const imageData = { width: 1, height: 1 } as ImageData;

            expect(
                toToolRunSSIMProps({
                    imageData,
                    template: rect(5, 5, 20, 20),
                    existingAnnotations: [rect(100, 100), { type: 'full_image' }],
                    autoMergeDuplicates: true,
                    shapeType: 'polygon',
                })
            ).toEqual({
                imageData,
                roi: { x: 0, y: 0, width: 1, height: 1 },
                template: { x: 5, y: 5, width: 20, height: 20, shapeType: 'rect' },
                existingAnnotations: [{ x: 100, y: 100, width: 10, height: 10, shapeType: 'rect' }],
                autoMergeDuplicates: true,
                shapeType: 'rect',
            });
        });
    });

    describe('convertToolMatchesToGetiMatches', () => {
        it('converts smart-tools rects to Geti rectangles and keeps the confidence', () => {
            expect(
                convertToolMatchesToGetiMatches([
                    { shape: { x: 1, y: 2, width: 3, height: 4, shapeType: 'rect' }, confidence: 0.8 },
                ])
            ).toEqual([match(rect(1, 2, 3, 4), 0.8)]);
        });
    });

    describe('filterSSIMResults', () => {
        const template = rect(0, 0);

        it('always keeps the template first', () => {
            expect(filterSSIMResults(imageBounds, [], template, [])).toEqual([match(template, 1)]);
        });

        it('drops matches that overlap the template, accepted matches or existing annotations', () => {
            const overlapsTemplate = match(rect(1, 1), 0.99);
            const accepted = match(rect(100, 100), 0.98);
            const overlapsAccepted = match(rect(101, 101), 0.97);
            const overlapsExisting = match(rect(501, 501), 0.96);
            const separate = match(rect(300, 300), 0.95);

            expect(
                filterSSIMResults(
                    imageBounds,
                    [overlapsTemplate, accepted, overlapsAccepted, overlapsExisting, separate],
                    template,
                    [rect(500, 500)]
                )
            ).toEqual([match(template, 1), accepted, separate]);
        });

        it('clamps matches that extend past the image', () => {
            const outside = match(rect(995, 995), 0.99);

            expect(filterSSIMResults(imageBounds, [outside], template, [])).toEqual([
                match(template, 1),
                match(rect(995, 995, 5, 5), 0.99),
            ]);
        });

        it(`caps the result at ${MAX_NUMBER_ITEMS} items, template included`, () => {
            const matches = Array.from({ length: MAX_NUMBER_ITEMS + 10 }, (_, index) =>
                match(rect((index % 50) * 20, 20 + Math.floor(index / 50) * 20), 0.99)
            );

            expect(filterSSIMResults(imageBounds, matches, template, [])).toHaveLength(MAX_NUMBER_ITEMS);
        });
    });

    describe('guessNumberOfItemsThreshold', () => {
        it(`counts the leading matches with a confidence of at least ${DEFAULT_CONFIDENCE_THRESHOLD}`, () => {
            const matches = [match(rect(0, 0), 1), match(rect(0, 0), 0.95), match(rect(0, 0), 0.8)];

            expect(guessNumberOfItemsThreshold(matches)).toBe(2);
        });

        it('returns every match when all are above the threshold', () => {
            expect(guessNumberOfItemsThreshold([match(rect(0, 0), 1), match(rect(0, 0), 0.9)])).toBe(2);
        });

        it('honours a custom threshold', () => {
            expect(guessNumberOfItemsThreshold([match(rect(0, 0), 1), match(rect(0, 0), 0.8)], 0.75)).toBe(2);
        });
    });

    describe('convertRectToShape', () => {
        it('keeps rectangles as is', () => {
            expect(convertRectToShape(rect(1, 2, 3, 4), 'rectangle')).toEqual(rect(1, 2, 3, 4));
        });

        it('converts rectangles to a clockwise polygon', () => {
            expect(convertRectToShape(rect(1, 2, 3, 4), 'polygon')).toEqual({
                type: 'polygon',
                points: [
                    { x: 1, y: 2 },
                    { x: 4, y: 2 },
                    { x: 4, y: 6 },
                    { x: 1, y: 6 },
                ],
            });
        });
    });
});
