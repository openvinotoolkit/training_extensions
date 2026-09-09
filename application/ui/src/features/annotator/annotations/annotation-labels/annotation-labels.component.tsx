// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { PointerEvent, useCallback } from 'react';

import { v4 as uuid } from 'uuid';

import { useLabelResolver } from '../../../../shared/annotator/labels';
import type { AnnotationLabel, AnnotationLabelRef } from '../../../../shared/types';
import { isPrediction } from '../utils';

import classes from './annotation-labels.module.scss';

const placeholderLabel = { id: uuid(), name: 'No label', color: 'var(--annotation-fill)', isPrediction: false };

// Screen-space dimensions for the foreignObject hit area
const LABEL_HEIGHT_PX = 24;
const LABEL_MAX_WIDTH_PX = 1000;

interface AnnotationLabelsProps {
    labels: AnnotationLabelRef[];
    onRemove: (labelId: string) => void;
    useBottomCorners?: boolean;
    isRemovable?: boolean;
}

const formatPredictionScore = (score: number) => {
    return new Intl.NumberFormat('en-US', { style: 'percent' }).format(score);
};

const getLabelText = (label: AnnotationLabel) => {
    return `${label.name} ${isPrediction(label) ? formatPredictionScore(label.probability) : ''}`.trim();
};

export const AnnotationLabels = ({
    labels,
    onRemove,
    useBottomCorners = false,
    isRemovable = true,
}: AnnotationLabelsProps) => {
    const { resolveAnnotationLabel } = useLabelResolver();

    const onDeleteLabel = useCallback(
        (labelId: string) => (event: PointerEvent) => {
            event.preventDefault();
            event.stopPropagation();
            onRemove(labelId);
        },
        [onRemove]
    );

    const resolvedLabels = labels.map(resolveAnnotationLabel).filter((label) => label !== undefined);
    const displayLabels = resolvedLabels.length ? resolvedLabels : [placeholderLabel];

    return (
        <foreignObject
            x={0}
            y={useBottomCorners ? 0 : -LABEL_HEIGHT_PX}
            width={LABEL_MAX_WIDTH_PX}
            height={LABEL_HEIGHT_PX}
            overflow='visible'
            className={useBottomCorners ? classes.labelsScalePolygon : classes.labelsScaleRect}
            aria-label={`Annotation labels`}
        >
            <div className={useBottomCorners ? classes.labelsContainerPolygon : classes.labelsContainerRect}>
                {displayLabels.map((label, index) => {
                    const isFirst = index === 0;
                    const isLast = index === displayLabels.length - 1;
                    const isPlaceholder = !resolvedLabels.length;

                    return (
                        <div
                            key={label.id}
                            className={classes.label}
                            style={{
                                '--label-color': label.color,
                                '--border-top-left': isFirst ? 'var(--spectrum-global-dimension-size-50)' : '0',
                                '--border-top-right': isLast ? 'var(--spectrum-global-dimension-size-50)' : '0',
                                '--border-bottom-left':
                                    useBottomCorners && isFirst ? 'var(--spectrum-global-dimension-size-50)' : '0',
                                '--border-bottom-right':
                                    useBottomCorners && isLast ? 'var(--spectrum-global-dimension-size-50)' : '0',
                            }}
                            aria-label={`label ${label.name} background`}
                        >
                            <span aria-label={`label ${label.name}`}>{getLabelText(label)}</span>
                            {!isPlaceholder && isRemovable && (
                                <button
                                    className={classes.removeButton}
                                    onPointerDown={onDeleteLabel(label.id)}
                                    aria-label={`Remove ${label.name}`}
                                >
                                    ×
                                </button>
                            )}
                        </div>
                    );
                })}
            </div>
        </foreignObject>
    );
};
