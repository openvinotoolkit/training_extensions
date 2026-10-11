// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from 'react';

import { isPolygonValid } from '@geti-ui/smart-tools/utils';

import { Annotation, Polygon } from '../../../../shared/types';
import { useAnnotationCommands } from '../../annotation-actions-provider.component';
import { AnnotationShapeRenderer } from '../../annotations/annotation-shape-renderer.component';
import { useSelectedMediaItem } from '../../selected-media-item-provider.component';
import { TranslateShape } from '../edit-bounding-box/translate-shape.component';
import { removeOffLimitPointsPolygon } from '../utils';
import { EditPoints } from './edit-points.component';

interface EditPolygonProps {
    zoom: number;
    annotation: Annotation & { shape: { type: 'polygon' } };
}

export const EditPolygon = ({ annotation, zoom }: EditPolygonProps) => {
    const { roi } = useSelectedMediaItem();
    const isAddPoint = useRef(false);
    const [shape, setShape] = useState(annotation.shape);
    const { updateAnnotations, deleteAnnotations } = useAnnotationCommands();

    useEffect(() => {
        setShape(annotation.shape);
    }, [annotation.shape]);

    // "removeOffLimitPoints" not only remove offlimit points but also in-between ones,
    // a new point is considered "in-between," and so it gets removed,
    // to avoid losing points we need not to use it when adding new ones
    const onComplete = (newShape: Polygon) => {
        const finalShape = isAddPoint.current ? newShape : removeOffLimitPointsPolygon(newShape, roi);

        if (isPolygonValid({ shapeType: 'polygon', points: finalShape.points })) {
            updateAnnotations([{ ...annotation, shape: finalShape }]);
        } else {
            deleteAnnotations([annotation.id]);
        }

        isAddPoint.current = false;
    };

    const translate = (inTranslate: { x: number; y: number }) => {
        setShape({
            ...shape,
            points: shape.points.map(({ x, y }) => ({
                x: x + inTranslate.x,
                y: y + inTranslate.y,
            })),
        });
    };

    const moveAnchorTo = (idx: number, x: number, y: number) => {
        isAddPoint.current = false;

        setShape((polygon) => ({
            ...shape,
            points: polygon.points.map((oldPoint, oldIdx) => {
                return idx === oldIdx ? { x, y } : oldPoint;
            }),
        }));
    };

    const addPoint = (idx: number, x: number, y: number) => {
        isAddPoint.current = true;

        setShape((polygon) => {
            const pointsBefore = [...polygon.points].splice(0, idx);
            const pointsAfter = [...polygon.points].splice(idx, polygon.points.length);
            const points = [...pointsBefore, { x, y }, ...pointsAfter];

            return {
                ...polygon,
                points,
            };
        });
    };

    const removePoints = (indexes: number[]): void => {
        const points = shape.points.filter((_, pointIdx) => !indexes.includes(pointIdx));

        setShape({ ...shape, points });
        onComplete({ ...shape, points });
    };

    return (
        <>
            <svg id={`translate-polygon-${annotation.id}`}>
                <TranslateShape
                    zoom={zoom}
                    translateShape={translate}
                    annotation={{ ...annotation, shape }}
                    onComplete={() => onComplete(shape)}
                >
                    <AnnotationShapeRenderer annotation={{ ...annotation, shape }} hideLabels />
                </TranslateShape>
            </svg>

            <svg data-testid={`edit-polygon-points-${annotation.id}`} id={`edit-polygon-points-${annotation.id}`}>
                <EditPoints
                    shape={shape}
                    zoom={zoom}
                    addPoint={addPoint}
                    removePoints={removePoints}
                    onComplete={() => onComplete(shape)}
                    moveAnchorTo={moveAnchorTo}
                />
            </svg>
        </>
    );
};
