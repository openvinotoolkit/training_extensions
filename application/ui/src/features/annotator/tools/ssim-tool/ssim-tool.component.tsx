// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from 'react';

import { useZoom } from '@/components/zoom/zoom.provider';

import { useAnnotationActions } from '../../../../shared/annotator/annotation-actions-provider.component';
import { getFormattedPoints } from '../../annotations/utils';
import { useAnnotatorLabels } from '../../annotator-labels-provider.component';
import { useSelectedMediaItem } from '../../selected-media-item-provider.component';
import { Rectangle } from '../../shapes/rectangle.component';
import { DEFAULT_ANNOTATION_STYLES } from '../../utils';
import { DrawingBox } from '../drawing-box-tool/drawing-box.component';
import { useAddAndSelectAnnotations } from '../use-add-and-select-annotations.hook';
import { useSSIM } from './use-ssim.hook';

import classes from './ssim-tool.module.scss';

export const SSIMTool = () => {
    const { scale: zoom } = useZoom();
    const { roi, image } = useSelectedMediaItem();
    const { annotations } = useAnnotationActions();
    const { selectedLabel } = useAnnotatorLabels();
    const { addAndSelectAnnotations } = useAddAndSelectAnnotations();
    const { runSSIM, reset, toolState, isProcessing, isLoading } = useSSIM();

    useEffect(() => {
        if (isProcessing || toolState.shapes.length === 0) {
            return;
        }

        const predictedShapes = toolState.shapes.slice(1);

        if (predictedShapes.length === 0) {
            reset();
            return;
        }

        addAndSelectAnnotations(predictedShapes, selectedLabel ? [selectedLabel] : []);
        reset();
    }, [toolState.shapes, isProcessing, addAndSelectAnnotations, selectedLabel, reset]);

    return (
        <>
            <svg
                aria-label='ssim preview'
                data-loading={isLoading}
                viewBox={`0 0 ${image.width} ${image.height}`}
                style={{ position: 'absolute', inset: 0, pointerEvents: 'none', overflow: 'visible' }}
            >
                {toolState.shapes.map((shape, index) => {
                    const isTemplate = index === 0;
                    const className = isTemplate ? classes.sourceArea : '';
                    const ariaLabel = isTemplate ? 'template' : 'prediction';

                    if (shape.type === 'rectangle') {
                        return (
                            <Rectangle
                                key={`ssim-${index}`}
                                rect={shape}
                                ariaLabel={ariaLabel}
                                styles={{ ...DEFAULT_ANNOTATION_STYLES, role: 'application', className }}
                            />
                        );
                    }

                    if (shape.type === 'polygon') {
                        return (
                            <polygon
                                key={`ssim-${index}`}
                                points={getFormattedPoints(shape.points)}
                                aria-label={ariaLabel}
                                className={className}
                                role='application'
                                fill={'var(--annotation-fill)'}
                                fillOpacity={'var(--annotation-fill-opacity)'}
                                stroke={'var(--annotation-stroke)'}
                                strokeLinecap={'round'}
                                strokeWidth={'calc(3px / var(--zoom-scale))'}
                                strokeOpacity={'var(--annotation-border-opacity, 1)'}
                            />
                        );
                    }

                    return null;
                })}
            </svg>

            <DrawingBox
                roi={roi}
                image={image}
                zoom={zoom}
                selectedLabel={selectedLabel}
                onComplete={(shapes) => {
                    const [template] = shapes;

                    if (template === undefined) {
                        return [];
                    }

                    runSSIM({
                        imageData: image,
                        roi,
                        template,
                        existingAnnotations: annotations.map((annotation) => annotation.shape),
                        autoMergeDuplicates: true,
                        shapeType: 'polygon',
                    });

                    return [];
                }}
            />
        </>
    );
};
