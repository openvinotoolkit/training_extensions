// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, useRef } from 'react';

import { clampBetween } from '@geti-ui/smart-tools/utils';
import { createUseGesture, dragAction, pinchAction, wheelAction } from '@use-gesture/react';

import type { Point } from './types';
import { useContainerSize } from './use-container-size';
import { usePanning } from './use-panning.hook';
import { Size, useSyncZoom } from './use-sync-zoom.hook';
import { useWheelPanning } from './use-wheel-panning.hook';
import { getZoomState } from './util';
import { useSetZoom, useZoom } from './zoom.provider';

import classes from './zoom.module.scss';

type ZoomTransformProps = {
    target: Size;
    children: ReactNode;
    zoomOutDivisor?: number;
    zoomInMultiplier?: number;
};

const useGesture = createUseGesture([wheelAction, pinchAction, dragAction]);

export const ZoomTransform = ({ children, target, zoomInMultiplier = 10, zoomOutDivisor = 2 }: ZoomTransformProps) => {
    const zoom = useZoom();
    const { setZoom } = useSetZoom();
    const { isPanning, setIsPanning } = usePanning();
    const containerRef = useRef<HTMLDivElement>(null);
    const containerSize = useContainerSize(containerRef);
    const { onPointerDown, onPointerUp, onPointerMove, onMouseLeave, isGrabbing } = useWheelPanning(setIsPanning);

    useSyncZoom({ container: containerSize, zoomInMultiplier, zoomOutDivisor, target });

    const cursorIcon = isPanning && isGrabbing ? 'grabbing' : isPanning ? 'grab' : 'default';

    useGesture(
        {
            onPinch: ({ origin, offset: [pinchScale] }) => {
                const rect = containerRef.current?.getBoundingClientRect();
                if (!rect) return;

                const cursorX = origin[0] - rect.left;
                const cursorY = origin[1] - rect.top;

                setZoom((prev) =>
                    getZoomState({
                        newScale: pinchScale,
                        cursorX,
                        cursorY,
                        initialCoordinates: prev.initialCoordinates,
                    })(prev)
                );
            },
            onWheel: ({ event, delta: [, verticalScrollDelta] }) => {
                // A trackpad pinch arrives as ctrl+wheel, which onPinch already handles
                if (event.ctrlKey) return;

                const rect = containerRef.current?.getBoundingClientRect();
                if (!rect) return;

                // Exponential so the step stays proportional and never flips sign on large deltas
                const factor = Math.exp(-verticalScrollDelta / 500);
                const cursorX = event.clientX - rect.left;
                const cursorY = event.clientY - rect.top;

                // Scaled off prev, since React batches the many events a trackpad emits per frame
                setZoom((prev) =>
                    getZoomState({
                        newScale: clampBetween(prev.initialCoordinates.scale, prev.scale * factor, prev.maxZoomIn),
                        cursorX,
                        cursorY,
                        initialCoordinates: prev.initialCoordinates,
                    })(prev)
                );
            },
            onDrag: ({ delta: [x, y] }) => handleTranslateUpdate({ x, y }),
        },
        {
            target: containerRef,
            eventOptions: { passive: false },
            wheel: { preventDefault: true },
            pinch: {
                preventDefault: true,
                // Makes offset[0] an absolute scale that resumes from the current zoom
                from: () => [zoom.scale, 0],
                scaleBounds: () => ({ min: zoom.initialCoordinates.scale, max: zoom.maxZoomIn }),
            },
            drag: { enabled: isPanning },
        }
    );

    const handleTranslateUpdate = ({ x, y }: Point) => {
        setZoom((prev) => ({
            ...prev,
            hasAnimation: false,
            translate: { x: prev.translate.x + x, y: prev.translate.y + y },
        }));
    };

    return (
        <div
            ref={containerRef}
            className={classes.wrapper}
            style={{
                cursor: cursorIcon,
                touchAction: 'none',
                transform: 'translate3d(0, 0, 0)',
                '--zoom-scale': zoom.scale,
            }}
            onPointerMove={onPointerMove(handleTranslateUpdate)}
            onPointerDown={onPointerDown}
            onPointerUp={onPointerUp}
            onMouseLeave={onMouseLeave}
        >
            <div
                data-testid='zoom-transform'
                className={classes.wrapperInternal}
                style={{
                    transformOrigin: '0 0',
                    transition: zoom.hasAnimation ? 'transform 0.2s ease' : 'none',
                    transform: `translate(${zoom.translate.x}px, ${zoom.translate.y}px) scale(${zoom.scale})`,
                }}
            >
                {children}
            </div>
        </div>
    );
};
