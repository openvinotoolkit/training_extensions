// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { InteractiveAnnotationPoint } from './segment-anything.interface';

// In screen pixels; divided by the zoom scale so points keep their size while zooming.
const RADIUS = 5;
const STROKE_WIDTH = 1.5;

type PromptPointsProps = {
    points: InteractiveAnnotationPoint[];
    zoomScale: number;
};

export const PromptPoints = ({ points, zoomScale }: PromptPointsProps) => {
    return (
        <g aria-label='Segment anything prompt points' pointerEvents='none'>
            {points.map((point) => (
                <circle
                    key={`${point.x}:${point.y}`}
                    aria-label={point.positive ? 'Positive point' : 'Negative point'}
                    cx={point.x}
                    cy={point.y}
                    r={RADIUS / zoomScale}
                    fill={point.positive ? 'var(--moss-tint-1)' : 'var(--coral-shade-1)'}
                    stroke='white'
                    strokeWidth={STROKE_WIDTH / zoomScale}
                />
            ))}
        </g>
    );
};
