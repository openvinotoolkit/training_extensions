// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import type { Annotation, Point } from '../../../../shared/types';
import { useAnnotationCommands } from '../../annotation-actions-provider.component';
import { AnnotationShapeRenderer } from '../../annotations/annotation-shape-renderer.component';
import { ANCHOR_SIZE, ResizeAnchor } from '../../resize-anchor.component';
import { useSelectedMediaItem } from '../../selected-media-item-provider.component';
import { getBoundingBoxInRoi, getBoundingBoxResizePoints, getClampedBoundingBox } from '../utils';
import { TranslateShape } from './translate-shape.component';

interface EditBoundingBoxProps {
    annotation: Annotation & { shape: { type: 'rectangle' } };
    zoom: number;
}

export const EditBoundingBox = ({ annotation, zoom }: EditBoundingBoxProps) => {
    const [shape, setShape] = useState(annotation.shape);
    const { roi } = useSelectedMediaItem();
    const { updateAnnotations } = useAnnotationCommands();

    const onComplete = () => {
        updateAnnotations([{ ...annotation, shape }]);
    };

    const translate = (point: Point) => {
        const newBoundingBox = getClampedBoundingBox(point, shape, roi);

        setShape({ ...shape, ...newBoundingBox });
    };

    const anchorPoints = getBoundingBoxResizePoints({
        gap: (2 * ANCHOR_SIZE) / zoom,
        boundingBox: shape,
        onResized: (boundingBox) => {
            setShape({ ...shape, ...getBoundingBoxInRoi(boundingBox, roi) });
        },
    });

    return (
        <>
            <TranslateShape
                zoom={zoom}
                annotation={{ ...annotation, shape }}
                translateShape={translate}
                onComplete={onComplete}
            >
                <AnnotationShapeRenderer annotation={{ ...annotation, shape }} />
            </TranslateShape>

            <g
                style={{ pointerEvents: 'auto' }}
                aria-label={`Edit bounding box points ${annotation.id}`}
                id={`edit-bounding-box-points-${annotation.id}`}
            >
                {anchorPoints.map((anchor) => {
                    return <ResizeAnchor key={anchor.label} zoom={zoom} onComplete={onComplete} {...anchor} />;
                })}
            </g>
        </>
    );
};
