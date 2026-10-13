// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Polygon } from '../../../../shared/types';
import { getFormattedPoints } from '../../annotations/utils';
import { ShapeStyle } from './utils';

type PolygonDrawProps = ShapeStyle<SVGPolygonElement> & {
    shape: Polygon;
    ariaLabel?: string;
    indicatorRadius?: number;
};

export const PolygonDraw = ({ shape, styles, indicatorRadius, className = '', ariaLabel = '' }: PolygonDrawProps) => {
    return (
        <g>
            <circle
                r={indicatorRadius}
                cx={shape.points[0].x}
                cy={shape.points[0].y}
                fill={'transparent'}
                stroke={`var(--energy-blue-shade)`}
                strokeWidth={styles?.strokeWidth}
            />
            <polyline
                {...styles}
                points={getFormattedPoints(shape.points)}
                className={className}
                aria-label={ariaLabel}
            />
        </g>
    );
};
