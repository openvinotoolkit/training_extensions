// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo } from 'react';

import type { ZoomState } from './types';
import { useSetZoom } from './zoom.provider';

export type Size = { width: number; height: number };

const DEFAULT_SCREEN_ZOOM = 0.9;
const getCenterCoordinates = (container: Size, target: Size): ZoomState['initialCoordinates'] => {
    // Scale image so that it fits perfectly in the container
    const scale = DEFAULT_SCREEN_ZOOM * Math.min(container.width / target.width, container.height / target.height);

    return {
        scale,
        // Center image, considering scale, transform origin is the top-left corner
        x: (container.width - target.width * scale) / 2,
        y: (container.height - target.height * scale) / 2,
    };
};

const INITIAL_ZOOM = { scale: 1.0, x: 0, y: 0 };

type useSyncZoomProps = { container: Size; target: Size; zoomInMultiplier: number };

export const useSyncZoom = ({ container, target, zoomInMultiplier }: useSyncZoomProps) => {
    const { setZoom } = useSetZoom();

    const targetZoom = useMemo(() => {
        if (container.width === undefined || container.height === undefined) {
            return INITIAL_ZOOM;
        }

        return getCenterCoordinates({ width: container.width, height: container.height }, target);
    }, [container, target]);

    useEffect(() => {
        const scale = Number(targetZoom.scale.toFixed(3));

        setZoom({
            scale,
            hasAnimation: false,
            maxZoomIn: scale * zoomInMultiplier,
            translate: {
                x: Number(targetZoom.x.toFixed(3)),
                y: Number(targetZoom.y.toFixed(3)),
            },
            initialCoordinates: { ...targetZoom },
        });
    }, [setZoom, zoomInMultiplier, targetZoom]);
};
