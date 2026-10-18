// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { PointerEvent, SVGProps } from 'react';

import { isLeftButton, isWheelButton } from '../../shared/buttons-utils';

type OnPointerDown = SVGProps<SVGElement>['onPointerDown'];
export const allowPanning = (onPointerDown?: OnPointerDown): OnPointerDown | undefined => {
    if (onPointerDown === undefined) {
        return;
    }

    return (event: PointerEvent<SVGElement>) => {
        const isPressingPanningHotKeys = (isLeftButton(event) && event.ctrlKey) || isWheelButton(event);

        if (isPressingPanningHotKeys) {
            return;
        }

        return onPointerDown(event);
    };
};

export const DEFAULT_ANNOTATION_STYLES = {
    fillOpacity: 'var(--annotation-fill-opacity)',
    fill: 'var(--annotation-fill)',
    stroke: 'var(--annotation-stroke)',
    strokeLinecap: 'round',
    strokeWidth: 'calc(3px / var(--zoom-scale))',
    strokeDashoffset: 0,
    strokeDasharray: 0,
    strokeOpacity: 'var(--annotation-border-opacity, 1)',
} satisfies SVGProps<SVGElement>;
