// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { PointerEvent, SVGProps } from 'react';

import type { Label } from '@/api/types';
import lassoClosingCursor from '@/assets/icons/cursor/lasso-closing.png';
import lassoDrawingCursor from '@/assets/icons/cursor/lasso-drawing.png';
import magneticLassoClosingCursor from '@/assets/icons/cursor/magnetic-lasso-closing.png';
import magneticLassoDrawingCursor from '@/assets/icons/cursor/magnetic-lasso-drawing.png';
import polygonClosingCursor from '@/assets/icons/cursor/polygon-closing.png';
import polygonDrawingCursor from '@/assets/icons/cursor/polygon-drawing.png';
import { isPolygonValid } from '@geti-ui/smart-tools/utils';
import { isEmpty, isNil, negate } from 'lodash-es';

import { isLeftButton } from '../../../../shared/buttons-utils';
import { Point, Polygon as PolygonType } from '../../../../shared/types';
import { DEFAULT_ANNOTATION_STYLES } from '../../utils';
import { PointerType } from '../utils';

export interface ShapeStyle<T> {
    styles?: SVGProps<T>;
    className?: string;
}

export enum PolygonMode {
    Lasso = 'lasso',
    LassoClose = 'lassoClose',
    Polygon = 'polygon',
    PolygonClose = 'polygonClose',
    MagneticLasso = 'magneticLasso',
    MagneticLassoClose = 'magneticLassoClose',
}

enum PointerIcons {
    Lasso = 'lasso-drawing',
    LassoClose = 'lasso-closing',
    Polygon = 'polygon-drawing',
    PolygonClose = 'polygon-closing',
    MagneticLasso = 'magnetic-lasso-drawing',
    MagneticLassoClose = 'magnetic-lasso-closing',
}

enum PointerIconsOffset {
    Lasso = '0 0',
    LassoClose = '0 0',
    Polygon = '0 0',
    PolygonClose = '0 0',
    MagneticLasso = '0 0',
    MagneticLassoClose = '0 0',
}

const TOOL_ICON: Record<PolygonMode, { icon: PointerIcons; offset: PointerIconsOffset; cursorUrl: string }> = {
    [PolygonMode.Lasso]: {
        icon: PointerIcons.Lasso,
        offset: PointerIconsOffset.Lasso,
        cursorUrl: lassoDrawingCursor,
    },
    [PolygonMode.Polygon]: {
        icon: PointerIcons.Polygon,
        offset: PointerIconsOffset.Polygon,
        cursorUrl: polygonDrawingCursor,
    },
    [PolygonMode.LassoClose]: {
        icon: PointerIcons.LassoClose,
        offset: PointerIconsOffset.LassoClose,
        cursorUrl: lassoClosingCursor,
    },
    [PolygonMode.MagneticLasso]: {
        icon: PointerIcons.MagneticLasso,
        offset: PointerIconsOffset.MagneticLasso,
        cursorUrl: magneticLassoDrawingCursor,
    },
    [PolygonMode.MagneticLassoClose]: {
        icon: PointerIcons.MagneticLassoClose,
        offset: PointerIconsOffset.MagneticLassoClose,
        cursorUrl: magneticLassoClosingCursor,
    },
    [PolygonMode.PolygonClose]: {
        icon: PointerIcons.PolygonClose,
        offset: PointerIconsOffset.PolygonClose,
        cursorUrl: polygonClosingCursor,
    },
};

export const START_POINT_FIELD_DEFAULT_RADIUS = 6;
export const START_POINT_FIELD_FOCUS_RADIUS = 8;

export const drawingStyles = (defaultLabel: Label | null): typeof DEFAULT_ANNOTATION_STYLES => {
    if (defaultLabel === null) {
        return DEFAULT_ANNOTATION_STYLES;
    }

    return {
        ...DEFAULT_ANNOTATION_STYLES,
        fill: defaultLabel.color,
        stroke: defaultLabel.color,
    };
};

export const getToolIcon = (polygonMode: PolygonMode | null) => {
    if (polygonMode === null) {
        return TOOL_ICON[PolygonMode.Polygon];
    }

    return TOOL_ICON[polygonMode];
};

type PointerSVGElement = PointerEvent<SVGElement>;
type CallbackPointerVoid = (event: PointerSVGElement) => void;

export const leftMouseButtonHandler =
    (callback: CallbackPointerVoid): CallbackPointerVoid =>
    (event: PointerSVGElement): void => {
        event.preventDefault();

        if (event.pointerType === PointerType.Touch || !isLeftButton(event)) return;

        callback(event);
    };

export const removeEmptySegments =
    (...newSegments: Point[][]) =>
    (prevSegments: Point[][]): Point[][] => {
        const validSegments = newSegments.filter(negate(isEmpty));

        return isEmpty(validSegments) ? [...prevSegments] : [...prevSegments, ...validSegments];
    };

export const isCloseMode = (mode: PolygonMode | null) => {
    if (mode == null) {
        return false;
    }

    return [PolygonMode.PolygonClose, PolygonMode.LassoClose, PolygonMode.MagneticLassoClose].includes(mode);
};

export const isPolygonReadyToClose = (currentPolygon: PolygonType | null): currentPolygon is PolygonType => {
    if (isNil(currentPolygon)) {
        return false;
    }

    const pointsWithoutLast = currentPolygon.points.slice(0, -1);

    return isPolygonValid({ shapeType: 'polygon', points: pointsWithoutLast });
};
