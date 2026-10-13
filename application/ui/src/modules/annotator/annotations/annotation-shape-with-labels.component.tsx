// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Key } from 'react';

import polylabel from 'polylabel';

import { EMPTY_LABEL_ID } from '../../../shared/labels';
import type { Annotation } from '../../../shared/types';
import { useAnnotationCommands, useIsAnnotatorReadOnly } from '../annotation-actions-provider.component';
import { useAnnotationVisibility } from '../annotation-visibility-provider.component';
import { useAnnotatorLabels } from '../annotator-labels-provider.component';
import { AnnotationLabels } from './annotation-labels/annotation-labels.component';
import { AnnotationShape } from './annotation-shape/annotation-shape.component';

type AnnotationShapeProps = {
    annotation: Annotation;
    hideLabels?: boolean;
};

type ShapeLabelsProps = {
    annotation: Annotation;
    useBottomCorners?: boolean;
};

// Owns the label-editing hooks so they're only subscribed to when labels are actually shown -
// tools that render shapes with hideLabels (e.g. edit-polygon) don't provide those contexts.
const ShapeLabels = ({ annotation, useBottomCorners = false }: ShapeLabelsProps) => {
    const { updateAnnotations } = useAnnotationCommands();
    const isReadOnlyMode = useIsAnnotatorReadOnly();
    const { selectedLabelId, setSelectedLabelId } = useAnnotatorLabels();

    const removeLabels = (labelId: Key | null) => {
        if (isReadOnlyMode) {
            return;
        }

        if (labelId === EMPTY_LABEL_ID && selectedLabelId === EMPTY_LABEL_ID) {
            setSelectedLabelId(null);
        }

        const updatedLabels = annotation.labels.filter((label) => label.id !== labelId);

        updateAnnotations([{ ...annotation, labels: updatedLabels }]);
    };

    return (
        <AnnotationLabels
            labels={annotation.labels}
            onRemove={removeLabels}
            useBottomCorners={useBottomCorners}
            isRemovable={!isReadOnlyMode}
        />
    );
};

export const AnnotationShapeWithLabels = ({ annotation, hideLabels = false }: AnnotationShapeProps) => {
    const { isVisible } = useAnnotationVisibility();
    const { shape } = annotation;

    if (shape.type === 'full_image') {
        return (
            <g display={isVisible ? 'block' : 'none'}>
                <AnnotationShape annotation={annotation} />
                {!hideLabels && <ShapeLabels annotation={annotation} />}
            </g>
        );
    }

    if (shape.type === 'rectangle') {
        return (
            <g transform={`translate(${shape.x}, ${shape.y})`} display={isVisible ? 'block' : 'none'}>
                <AnnotationShape annotation={{ ...annotation, shape: { ...shape, x: 0, y: 0 } }} />
                {!hideLabels && <ShapeLabels annotation={annotation} />}
            </g>
        );
    }

    if (hideLabels) {
        return <AnnotationShape annotation={annotation} />;
    }

    const polygonCoords = [shape.points.map((point) => [point.x, point.y])];
    const [labelX, labelY] = polylabel(polygonCoords);

    return (
        <g transform={`translate(${labelX}, ${labelY})`} display={isVisible ? 'block' : 'none'}>
            <g transform={`translate(${-labelX}, ${-labelY})`}>
                <AnnotationShape annotation={annotation} />
            </g>
            <ShapeLabels annotation={annotation} useBottomCorners />
        </g>
    );
};
