// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Point } from '../../../../shared/types';
import type { InteractiveAnnotationPoint } from './segment-anything.interface';

const findPointIndexNear = (points: InteractiveAnnotationPoint[], target: Point, radius: number): number => {
    let closestIndex = -1;
    let closestDistance = radius;

    points.forEach((point, index) => {
        const distance = Math.hypot(point.x - target.x, point.y - target.y);

        if (distance <= closestDistance) {
            closestIndex = index;
            closestDistance = distance;
        }
    });

    return closestIndex;
};

// Clicking on an existing prompt point removes it, clicking anywhere else adds a new one.
export const togglePromptPoint = (
    points: InteractiveAnnotationPoint[],
    target: Point,
    positive: boolean,
    hitRadius: number
): InteractiveAnnotationPoint[] => {
    const index = findPointIndexNear(points, target, hitRadius);

    if (index !== -1) {
        return points.filter((_, pointIndex) => pointIndex !== index);
    }

    return [...points, { x: target.x, y: target.y, positive }];
};
