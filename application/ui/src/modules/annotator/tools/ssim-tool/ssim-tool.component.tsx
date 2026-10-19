// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from 'react';

import { toast } from '@/components/toast/toast.component';
import { useZoom } from '@/components/zoom/zoom.provider';
import { useTranslation } from '@/i18n';

import { useAnnotations } from '../../annotation-document-provider.component';
import { getFormattedPoints } from '../../annotations/utils';
import { useAnnotatorLabels } from '../../annotator-labels-provider.component';
import { useSelectedMediaItem } from '../../selected-media-item-provider.component';
import { Rectangle } from '../../shapes/rectangle.component';
import { DEFAULT_ANNOTATION_STYLES } from '../../utils';
import { DrawingBox } from '../drawing-box-tool/drawing-box.component';
import { ToolLoading } from '../tool-loading.component';
import { useSSIMState } from './ssim-state-provider.component';

import classes from './ssim-tool.module.scss';

export const SSIMTool = () => {
    const { t } = useTranslation();
    const { scale: zoom } = useZoom();
    const { roi, image } = useSelectedMediaItem();
    const { annotations } = useAnnotations();
    const { selectedLabel } = useAnnotatorLabels();
    const { runSSIM, toolState, isProcessing, isLoading, isError, error, autoMergeDuplicates } = useSSIMState();

    useEffect(() => {
        if (!isError) {
            return;
        }

        toast({
            id: 'ssim-tool-error',
            type: 'error',
            message: t('annotator.tools.ssim.error', {
                message: error?.message ?? t('annotator.tools.autoSegmentation.unknownError'),
            }),
        });
    }, [isError, error, t]);

    if (isLoading) {
        return <ToolLoading message={t('annotator.tools.ssim.loading')} />;
    }

    return (
        <>
            <svg
                aria-label='ssim preview'
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
                // A second run would overwrite the in-flight one's results before they are added.
                isDisabled={isProcessing}
                onComplete={(shapes) => {
                    const [template] = shapes;

                    if (template === undefined) {
                        return [];
                    }

                    runSSIM({
                        imageData: image,
                        template,
                        existingAnnotations: annotations.map((annotation) => annotation.shape),
                        autoMergeDuplicates,
                    });

                    return [];
                }}
            />
        </>
    );
};
