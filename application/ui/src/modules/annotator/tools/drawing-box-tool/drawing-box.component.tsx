// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { PointerEvent, useRef, useState } from 'react';

import type { Label } from '@/api/types';
import selectionCursor from '@/assets/icons/selection.svg?url';
import { clampBox, clampPointBetweenImage, pointsToRect } from '@geti-ui/smart-tools/utils';
import { useEventListener } from 'hooks/event-listener.hook';

import { isLeftButton } from '../../../../shared/buttons-utils';
import type { Point, Rect as RectInterface, RegionOfInterest } from '../../../../shared/types';
import { Rectangle } from '../../shapes/rectangle.component';
import { DEFAULT_ANNOTATION_STYLES } from '../../utils';
import { SvgToolCanvas } from '../svg-tool-canvas.component';
import { getRelativePoint, PointerType } from '../utils';
import { Crosshair } from './crosshair/crosshair.component';
import { useCrosshair } from './crosshair/use-crosshair.hook';

const CURSOR_OFFSET = '7 8';
interface DrawingBoxInterface {
    onComplete: (shapes: RectInterface[], labels: Label[]) => string[];
    roi: RegionOfInterest;
    image: ImageData;
    selectedLabel: Label | null;
    zoom: number;
    isDisabled?: boolean;
}

export const DrawingBox = ({
    roi,
    zoom,
    image,
    selectedLabel,
    onComplete,
    isDisabled = false,
}: DrawingBoxInterface) => {
    const [startPoint, setStartPoint] = useState<Point | null>(null);
    const [boundingBox, setBoundingBox] = useState<RectInterface | null>(null);

    const ref = useRef<SVGRectElement>(null);

    const clampPoint = clampPointBetweenImage(image);
    const crosshair = useCrosshair(ref, zoom);

    const onPointerMove = (event: PointerEvent<SVGSVGElement>): void => {
        crosshair.onPointerMove(event);

        if (ref.current === null) {
            return;
        }

        if (startPoint === null || !event.currentTarget.hasPointerCapture(event.pointerId)) {
            return;
        }

        const endPoint = clampPoint(getRelativePoint(ref.current, { x: event.clientX, y: event.clientY }, zoom));

        setBoundingBox({ type: 'rectangle', ...clampBox(pointsToRect(startPoint, endPoint), roi) });
    };

    const onPointerDown = (event: PointerEvent<SVGSVGElement>): void => {
        if (isDisabled || startPoint !== null || ref.current === null) {
            return;
        }

        const button = {
            button: event.button,
            buttons: event.buttons,
        };

        if (event.pointerType === PointerType.Touch || !isLeftButton(button)) {
            return;
        }

        const mouse = clampPoint(getRelativePoint(ref.current, { x: event.clientX, y: event.clientY }, zoom));

        event.currentTarget.setPointerCapture(event.pointerId);

        setStartPoint(mouse);
        setBoundingBox({ type: 'rectangle', x: mouse.x, y: mouse.y, width: 0, height: 0 });
    };

    const onPointerUp = (event: PointerEvent<SVGSVGElement>): void => {
        if (event.pointerType === PointerType.Touch) {
            return;
        }

        if (boundingBox === null) {
            return;
        }

        // Don't make empty annotations
        if (boundingBox.width > 1 && boundingBox.height > 1) {
            onComplete([boundingBox], selectedLabel ? [selectedLabel] : []);
        }

        setCleanState();

        event.currentTarget.releasePointerCapture(event.pointerId);
    };

    const setCleanState = () => {
        setStartPoint(null);
        setBoundingBox(null);
    };

    useEventListener('keydown', (event: KeyboardEvent) => {
        if (event.key === 'Escape') {
            setCleanState();
        }
    });

    return (
        <SvgToolCanvas
            image={image}
            canvasRef={ref}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerDown={onPointerDown}
            onPointerLeave={crosshair.onPointerLeave}
            style={{ cursor: isDisabled ? 'progress' : `url(${selectionCursor}) ${CURSOR_OFFSET}, auto` }}
        >
            {boundingBox ? (
                <Rectangle
                    ariaLabel={'bounding box'}
                    rect={boundingBox}
                    styles={{ role: 'application', ...DEFAULT_ANNOTATION_STYLES }}
                />
            ) : null}
            <Crosshair location={crosshair.location} zoom={zoom} />
        </SvgToolCanvas>
    );
};
