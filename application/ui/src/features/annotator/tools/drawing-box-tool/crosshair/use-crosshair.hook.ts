// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState, type PointerEvent, type RefObject } from 'react';

import type { Point } from '../../../../../shared/types';
import { getRelativePoint } from '../../utils';

interface UseCrosshair {
    location: Point | null;
    onPointerMove: (event: PointerEvent<SVGSVGElement>) => void;
    onPointerLeave: (event: PointerEvent<SVGSVGElement>) => void;
}

export const useCrosshair = (canvasRef: RefObject<SVGRectElement | null>, zoom: number): UseCrosshair => {
    const [location, setLocation] = useState<Point | null>(null);

    const onPointerMove = (event: PointerEvent<SVGSVGElement>) => {
        if (canvasRef.current === null) {
            return;
        }

        const newLocation = getRelativePoint(canvasRef.current, { x: event.clientX, y: event.clientY }, zoom);

        setLocation(newLocation);
    };

    const onPointerLeave = (_event: PointerEvent<SVGSVGElement>) => {
        setLocation(null);
    };

    return { location, onPointerMove, onPointerLeave };
};
