// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useRef, useState } from 'react';

import type { Media } from '@/api/types';

import type { Point, RegionOfInterest, Shape } from '../../../../shared/types';
import { removeOffLimitPoints } from '../utils';
import type { InteractiveAnnotationPoint, SegmentAnythingDecodingResult } from './segment-anything.interface';
import { useSingleStackFn } from './use-single-stack-fn.hook';
import { useWithCancel } from './use-with-cancel';
import { togglePromptPoint } from './utils';

interface UseSegmentAnythingPromptProps {
    decodingQueryFn: (
        points: InteractiveAnnotationPoint[],
        previousMask?: Float32Array
    ) => Promise<SegmentAnythingDecodingResult>;
    roi: RegionOfInterest;
    // Points belong to the image they were placed on, so changing it clears the prompt.
    mediaItem: Media;
}

// The preview follows the cursor until points are placed, then it follows the points.
export const useSegmentAnythingPrompt = ({ decodingQueryFn, roi, mediaItem }: UseSegmentAnythingPromptProps) => {
    const [previewShapes, setPreviewShapes] = useState<Shape[]>([]);
    const [points, setPoints] = useState<InteractiveAnnotationPoint[]>([]);
    // Decoded from the current `points`; fed back to the decoder to refine the object when a point is added.
    const [refinementMask, setRefinementMask] = useState<Float32Array>();
    const [isDecoding, setIsDecoding] = useState(false);
    // Counter (not boolean) because calls overlap and intermediate ones get cancelled.
    const pendingDecodesRef = useRef(0);

    const throttledDecodingQueryFn = useSingleStackFn(decodingQueryFn);
    const cancellableDecodingQueryFn = useWithCancel(throttledDecodingQueryFn);

    const [promptMediaItem, setPromptMediaItem] = useState(mediaItem);
    if (promptMediaItem !== mediaItem) {
        setPromptMediaItem(mediaItem);
        setPoints([]);
        setRefinementMask(undefined);
        setPreviewShapes([]);
    }

    // Resolves with the decoded low resolution mask. Rejects when superseded by a newer call or cancelled.
    const decode = async (prompt: InteractiveAnnotationPoint[], previousMask?: Float32Array) => {
        pendingDecodesRef.current += 1;
        if (!isDecoding) setIsDecoding(true);

        try {
            const { shapes, lowResMask } = await cancellableDecodingQueryFn.call(prompt, previousMask);

            setPreviewShapes(shapes.map((shape) => removeOffLimitPoints(shape, roi)));

            return lowResMask;
        } finally {
            pendingDecodesRef.current = Math.max(0, pendingDecodesRef.current - 1);
            if (pendingDecodesRef.current === 0) {
                setIsDecoding(false);
            }
        }
    };

    const cancelDecoding = () => {
        cancellableDecodingQueryFn.cancel();
        pendingDecodesRef.current = 0;
        setIsDecoding(false);
    };

    const clear = () => {
        cancelDecoding();
        setPoints([]);
        setRefinementMask(undefined);
        setPreviewShapes([]);
    };

    const hover = (point: Point) => {
        if (points.length > 0) {
            return;
        }

        decode([{ ...point, positive: true }]).catch(() => undefined);
    };

    const clearHover = () => {
        if (points.length > 0) {
            return;
        }

        cancelDecoding();
        setPreviewShapes([]);
    };

    const togglePoint = (point: Point, positive: boolean, hitRadius: number) => {
        const nextPoints = togglePromptPoint(points, point, positive, hitRadius);

        if (nextPoints.length === 0) {
            clear();

            return;
        }

        // After removing a point the mask still reflects it, so the object is predicted from scratch.
        const isAddingPoint = points.length > 0 && nextPoints.length > points.length;

        setPoints(nextPoints);
        setRefinementMask(undefined);

        decode(nextPoints, isAddingPoint ? refinementMask : undefined)
            .then((mask) => setRefinementMask(mask))
            .catch(() => undefined);
    };

    return { points, previewShapes, isDecoding, hover, clearHover, togglePoint, clear };
};
