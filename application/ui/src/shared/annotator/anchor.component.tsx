// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ComponentRef, CSSProperties, PointerEvent, ReactNode, useRef, useState } from 'react';

import { CursorContextMenu } from '@/components/cursor-context-menu/cursor-context-menu.component';
import { useOverlayTriggerState } from '@react-stately/overlays';
import { isFunction } from 'lodash-es';

import { isLeftButton } from '../buttons-utils';
import { Point } from '../types';

interface AnchorProps {
    children: ReactNode;
    x: number;
    y: number;
    size: number;
    zoom: number;
    label: string;
    fill?: string;
    contextMenu?: (onClose: () => void) => ReactNode;
    cursor?: CSSProperties['cursor'];
    onStart?: () => void;
    onComplete: () => void;
    moveAnchorTo: (x: number, y: number) => void;
}

export const Anchor = ({
    x,
    y,
    fill = 'white',
    size,
    zoom,
    label,
    cursor,
    children,
    contextMenu,
    onStart,
    moveAnchorTo,
    onComplete,
}: AnchorProps) => {
    const state = useOverlayTriggerState({});
    const triggerRef = useRef<ComponentRef<'svg'>>(null);
    const [dragFrom, setDragFrom] = useState<Point | null>(null);

    const onPointerDown = (event: PointerEvent) => {
        event.preventDefault();

        if (event.pointerType === 'touch' || !isLeftButton(event)) {
            return;
        }

        event.currentTarget.setPointerCapture(event.pointerId);

        const mouse = { x: Math.round(event.clientX / zoom), y: Math.round(event.clientY / zoom) };

        isFunction(onStart) && onStart();
        setDragFrom({ x: mouse.x - x, y: mouse.y - y });
    };

    const onPointerMove = (event: PointerEvent) => {
        event.preventDefault();

        if (dragFrom === null) {
            return;
        }

        const mouse = { x: Math.round(event.clientX / zoom), y: Math.round(event.clientY / zoom) };

        moveAnchorTo(mouse.x - dragFrom.x, mouse.y - dragFrom.y);
    };

    const onPointerUp = (event: PointerEvent) => {
        if (event.pointerType === 'touch' || !isLeftButton(event)) {
            return;
        }

        event.preventDefault();
        event.currentTarget.releasePointerCapture(event.pointerId);

        setDragFrom(null);
        onComplete();
    };

    // We render both a visual anchor and an invisible anchor that has a larger
    // clicking area than the visible one
    const interactiveAnchorProps = {
        style: { cursor },
        stroke: 'none',
        fill: dragFrom === null ? fill : 'var(--energy-blue)',
        'aria-label': label,
        onPointerUp,
        onPointerMove,
        onPointerDown,
    };

    return (
        <g ref={triggerRef} data-resize-anchor='true'>
            {children}
            <rect
                x={x - size}
                y={y - size}
                cx={x}
                cy={y}
                width={size * 2}
                height={size * 2}
                fillOpacity={0}
                {...interactiveAnchorProps}
            />

            <CursorContextMenu state={state} triggerRef={triggerRef} onOpen={state.open}>
                {isFunction(contextMenu) && contextMenu(state.close)}
            </CursorContextMenu>
        </g>
    );
};
