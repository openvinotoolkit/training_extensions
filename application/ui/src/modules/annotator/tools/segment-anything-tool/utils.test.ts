// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { togglePromptPoint } from './utils';

describe('togglePromptPoint', () => {
    it('appends a positive or negative point when clicking on an empty area', () => {
        const points = togglePromptPoint([], { x: 10, y: 10 }, true, 5);

        expect(points).toEqual([{ x: 10, y: 10, positive: true }]);
        expect(togglePromptPoint(points, { x: 50, y: 50 }, false, 5)).toEqual([
            { x: 10, y: 10, positive: true },
            { x: 50, y: 50, positive: false },
        ]);
    });

    it('removes an existing point when clicking within the hit radius, regardless of its polarity', () => {
        const points = [
            { x: 10, y: 10, positive: true },
            { x: 50, y: 50, positive: false },
        ];

        expect(togglePromptPoint(points, { x: 53, y: 54 }, true, 5)).toEqual([{ x: 10, y: 10, positive: true }]);
    });

    it('removes only the closest point when several are within the hit radius', () => {
        const points = [
            { x: 10, y: 10, positive: true },
            { x: 14, y: 10, positive: false },
        ];

        expect(togglePromptPoint(points, { x: 13, y: 10 }, true, 5)).toEqual([{ x: 10, y: 10, positive: true }]);
    });

    it('does not mutate the given points', () => {
        const points = [{ x: 10, y: 10, positive: true }];

        togglePromptPoint(points, { x: 10, y: 10 }, true, 5);
        togglePromptPoint(points, { x: 90, y: 90 }, true, 5);

        expect(points).toEqual([{ x: 10, y: 10, positive: true }]);
    });
});
