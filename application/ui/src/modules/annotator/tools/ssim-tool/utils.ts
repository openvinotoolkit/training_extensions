// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { RunSSIMProps as ToolRunSSIMProps, SSIMMatch as ToolSSIMMatch } from '@geti-ui/smart-tools';
import { clampBox, roiFromImage } from '@geti-ui/smart-tools/utils';

import type { Rect, RegionOfInterest, Shape } from '../../../../shared/types';
import { convertToolShapeToGetiShape, getBoundingRectFromShape, intersectionOverUnion } from '../utils';

// Upper bound on matches (template included), so a weak template can't flood the image.
export const MAX_NUMBER_ITEMS = 100;
// Scores are min-max normalised per run, so this is relative to the template's own score (1.0).
export const DEFAULT_CONFIDENCE_THRESHOLD = 0.9;

export type SSIMMatch = Omit<ToolSSIMMatch, 'shape'> & {
    shape: Rect;
};

export type SSIMShapeType = Extract<Shape['type'], 'rectangle' | 'polygon'>;

export type RunSSIMProps = Omit<ToolRunSSIMProps, 'roi' | 'template' | 'existingAnnotations' | 'shapeType'> & {
    template: Rect;
    existingAnnotations: Shape[];
    shapeType: SSIMShapeType;
};

const toToolRect = (rect: Rect): ToolRunSSIMProps['template'] => {
    const { x, y, width, height } = rect;

    return {
        x,
        y,
        width,
        height,
        shapeType: 'rect',
    };
};

export const getExistingRects = (shapes: Shape[]): Rect[] => {
    return shapes.map(getBoundingRectFromShape).filter((shape): shape is Rect => shape !== null);
};

export const toToolRunSSIMProps = ({
    imageData,
    template,
    existingAnnotations,
    autoMergeDuplicates,
}: RunSSIMProps): ToolRunSSIMProps => {
    return {
        imageData,
        roi: roiFromImage(imageData),
        template: toToolRect(template),
        existingAnnotations: getExistingRects(existingAnnotations).map(toToolRect),
        autoMergeDuplicates,
        shapeType: 'rect',
    };
};

export const convertToolMatchesToGetiMatches = (matches: ToolSSIMMatch[]): SSIMMatch[] => {
    return matches.map((match) => ({
        ...match,
        shape: convertToolShapeToGetiShape(match.shape),
    }));
};

export const filterSSIMResults = (
    imageBounds: RegionOfInterest,
    items: SSIMMatch[],
    template: Rect,
    filter: Rect[],
    maxItems = MAX_NUMBER_ITEMS,
    overlapThreshold = 0.2
): SSIMMatch[] => {
    const collector: SSIMMatch[] = [{ shape: template, confidence: 1 }];
    const filterAsMatches = filter.map((shape) => ({ shape, confidence: 1 }));
    // smart-tools' downscaling and half-pixel offset push matches at the right/bottom edge past the image.
    const clampedItems = items.map((item) => ({
        ...item,
        shape: { type: 'rectangle' as const, ...clampBox(item.shape, imageBounds) },
    }));

    for (const value of clampedItems) {
        if (collector.length >= maxItems) {
            break;
        }

        const overlapsWithExisting = [...filterAsMatches, ...collector].some(
            (otherMatch) => intersectionOverUnion(otherMatch.shape, value.shape) > overlapThreshold
        );

        if (!overlapsWithExisting) {
            collector.push(value);
        }
    }

    return collector;
};

export const guessNumberOfItemsThreshold = (
    matches: SSIMMatch[],
    confidenceThreshold = DEFAULT_CONFIDENCE_THRESHOLD
): number => {
    const guess = matches.findIndex(({ confidence }) => confidence < confidenceThreshold);

    return guess === -1 ? matches.length : guess;
};

export const convertRectToShape = (rectangle: Rect, shapeType: SSIMShapeType): Shape => {
    if (shapeType === 'rectangle') {
        return rectangle;
    }

    return {
        type: 'polygon',
        points: [
            { x: rectangle.x, y: rectangle.y },
            { x: rectangle.x + rectangle.width, y: rectangle.y },
            { x: rectangle.x + rectangle.width, y: rectangle.y + rectangle.height },
            { x: rectangle.x, y: rectangle.y + rectangle.height },
        ],
    };
};
