// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { PointerEvent, useEffect, useRef } from 'react';

import selectionCursor from '@/assets/icons/selection.svg?url';
import { toast } from '@/components/toast/toast.component';
import { useZoom } from '@/components/zoom/zoom.provider';
import { useTranslation } from '@/i18n';
import { clampPointBetweenImage } from '@geti-ui/smart-tools/utils';
import { useGetDatasetMediaItems } from 'hooks/use-get-dataset-media-items.hook';

import { HOTKEYS } from '../../../../shared/hotkeys-definition';
import type { Annotation, Point, RegionOfInterest } from '../../../../shared/types';
import { AnnotationShape } from '../../annotations/annotation-shape/annotation-shape.component';
import { MaskAnnotations } from '../../annotations/mask-annotations.component';
import { useAnnotatorLabels } from '../../annotator-labels-provider.component';
import { useSelectedMediaItem } from '../../selected-media-item-provider.component';
import { useNextMediaItem } from '../../shell/utils';
import { SvgToolCanvas } from '../svg-tool-canvas.component';
import { ToolLoading } from '../tool-loading.component';
import { useAddAndSelectAnnotations } from '../use-add-and-select-annotations.hook';
import { useToolResultHotkey } from '../use-tool-result-hotkey.hook';
import { getRelativePoint } from '../utils';
import { PromptPoints } from './prompt-points.component';
import { useSegmentAnythingPrompt } from './use-segment-anything-prompt.hook';
import { useSegmentAnythingModel } from './use-segment-anything.hook';

import classes from './segment-anything.module.scss';

interface PreviewAnnotationsProps {
    previewAnnotations: Annotation[];
    image: Pick<RegionOfInterest, 'width' | 'height'>;
}

const CURSOR_OFFSET = '7 8';
// In screen pixels, so placed points are equally easy to hit at any zoom level.
const PROMPT_POINT_HIT_RADIUS = 8;

const PreviewAnnotations = ({ previewAnnotations, image }: PreviewAnnotationsProps) => {
    if (previewAnnotations.length === 0) return null;

    return (
        <MaskAnnotations isEnabled annotations={previewAnnotations} width={image.width} height={image.height}>
            {previewAnnotations.map((annotation) => (
                <g
                    key={annotation.id}
                    aria-label='Segment anything preview'
                    stroke={'var(--energy-blue-shade)'}
                    strokeWidth={'calc(3px / var(--zoom-scale))'}
                    fill={'transparent'}
                    fillOpacity={'var(--annotation-fill-opacity)'}
                    className={classes.animateStroke}
                >
                    <AnnotationShape annotation={annotation} />
                </g>
            ))}
        </MaskAnnotations>
    );
};

export const SegmentAnythingTool = () => {
    const ref = useRef<SVGSVGElement>(null);
    const canvasRef = useRef<SVGRectElement>(null);

    const { t } = useTranslation();
    const zoom = useZoom();
    const { roi, image, mediaItem } = useSelectedMediaItem();
    const { items } = useGetDatasetMediaItems();
    const nextMediaItem = useNextMediaItem(mediaItem, items);
    const { selectedLabel } = useAnnotatorLabels();
    const { addAndSelectAnnotations } = useAddAndSelectAnnotations();
    const { isLoading, isError, error, decodingQueryFn } = useSegmentAnythingModel({ nextMediaItem });
    const prompt = useSegmentAnythingPrompt({ decodingQueryFn, roi, mediaItem });

    const clampPoint = clampPointBetweenImage(image);

    const getImagePoint = (event: PointerEvent<SVGSVGElement>): Point | undefined => {
        if (!canvasRef.current) {
            return undefined;
        }

        return clampPoint(getRelativePoint(canvasRef.current, { x: event.clientX, y: event.clientY }, zoom.scale));
    };

    const handlePointerMove = (event: PointerEvent<SVGSVGElement>) => {
        const point = getImagePoint(event);

        if (point !== undefined) {
            prompt.hover(point);
        }
    };

    const handlePointerUp = (event: PointerEvent<SVGSVGElement>) => {
        if (!ref.current) {
            return;
        }

        if (event.button !== 0 && event.button !== 2) {
            return;
        }

        // Shift + click includes an area, Alt + click excludes it; a plain click accepts the preview.
        if (event.shiftKey || event.altKey) {
            const point = getImagePoint(event);

            if (point !== undefined) {
                prompt.togglePoint(point, !event.altKey, PROMPT_POINT_HIT_RADIUS / zoom.scale);
            }

            return;
        }

        if (prompt.previewShapes.length === 0) {
            return;
        }

        addAndSelectAnnotations(prompt.previewShapes, selectedLabel ? [selectedLabel] : []);
        prompt.clear();
    };

    useToolResultHotkey(HOTKEYS.rejectToolResult, prompt.clear, prompt.points.length > 0);

    const previewAnnotations = prompt.previewShapes.map((shape, idx): Annotation => {
        return {
            shape,
            labels: selectedLabel ? [{ id: selectedLabel.id }] : [],
            id: `${idx}`,
        };
    });

    useEffect(() => {
        if (!isError) {
            return;
        }

        toast({
            id: 'segment-anything-tool-error',
            type: 'error',
            message: t('annotator.tools.autoSegmentation.error', {
                message: error?.message ?? t('annotator.tools.autoSegmentation.unknownError'),
            }),
        });
    }, [isError, error, t]);

    if (isLoading) {
        return <ToolLoading message={t('annotator.tools.autoSegmentation.loading')} />;
    }

    return (
        <SvgToolCanvas
            ref={ref}
            image={image}
            canvasRef={canvasRef}
            aria-label='SAM tool canvas'
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerLeave={prompt.clearHover}
            style={{
                cursor: prompt.isDecoding ? 'progress' : `url(${selectionCursor}) ${CURSOR_OFFSET}, auto`,
            }}
        >
            <PreviewAnnotations previewAnnotations={previewAnnotations} image={image} />
            <PromptPoints points={prompt.points} zoomScale={zoom.scale} />
        </SvgToolCanvas>
    );
};
