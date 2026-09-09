// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { PointerEvent, useEffect, useRef, useState } from 'react';

import { toast } from '@/components/toast/toast.component';
import { useZoom } from '@/components/zoom/zoom.provider';
import { clampPointBetweenImage } from '@geti-ui/smart-tools/utils';
import { useGetDatasetMediaItems } from 'hooks/use-get-dataset-media-items.hook';

import selectionCursor from '../../../../assets/icons/selection.svg?url';
import type { Annotation, RegionOfInterest, Shape } from '../../../../shared/types';
import { useNextMediaItem } from '../../../dataset/media-preview/utils';
import { AnnotationShape } from '../../annotations/annotation-shape/annotation-shape.component';
import { MaskAnnotations } from '../../annotations/mask-annotations.component';
import { useAnnotatorLabels } from '../../annotator-labels-provider.component';
import { useSelectedMediaItem } from '../../selected-media-item-provider.component';
import { SvgToolCanvas } from '../svg-tool-canvas.component';
import { useAddAndSelectAnnotations } from '../use-add-and-select-annotations.hook';
import { getRelativePoint, removeOffLimitPoints } from '../utils';
import { SAMLoading } from './sam-loading.component';
import { useSegmentAnythingModel } from './use-segment-anything.hook';
import { useSingleStackFn } from './use-single-stack-fn.hook';
import { useWithCancel } from './use-with-cancel';

import classes from './segment-anything.module.scss';

interface PreviewAnnotationsProps {
    previewAnnotations: Annotation[];
    image: Pick<RegionOfInterest, 'width' | 'height'>;
}

const CURSOR_OFFSET = '7 8';

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
    const [previewShapes, setPreviewShapes] = useState<Shape[]>([]);
    const [isDecoding, setIsDecoding] = useState(false);
    const ref = useRef<SVGSVGElement>(null);

    const zoom = useZoom();
    const { roi, image, mediaItem } = useSelectedMediaItem();
    const { items } = useGetDatasetMediaItems();
    const nextMediaItem = useNextMediaItem(mediaItem, items);
    const { selectedLabel } = useAnnotatorLabels();
    const { addAndSelectAnnotations } = useAddAndSelectAnnotations();
    const { isLoading, isError, error, decodingQueryFn } = useSegmentAnythingModel({ nextMediaItem });
    const throttledDecodingQueryFn = useSingleStackFn(decodingQueryFn);
    const cancellableThrottledDecodingQueryFn = useWithCancel(throttledDecodingQueryFn);

    const canvasRef = useRef<SVGRectElement>(null);
    const hasShownErrorToastRef = useRef(false);
    // Counter (not boolean) because pointer moves overlap and we cancel
    // intermediate ones — the cursor stays busy while ANY call is pending.
    const pendingDecodesRef = useRef(0);

    const clampPoint = clampPointBetweenImage(image);

    const handleMouseMove = (event: PointerEvent<SVGSVGElement>) => {
        if (!canvasRef.current) {
            return;
        }

        const point = clampPoint(
            getRelativePoint(canvasRef.current, { x: event.clientX, y: event.clientY }, zoom.scale)
        );

        pendingDecodesRef.current += 1;
        if (!isDecoding) setIsDecoding(true);

        cancellableThrottledDecodingQueryFn
            .call([{ ...point, positive: true }])
            .then((shapes) => {
                setPreviewShapes(shapes.map((shape) => removeOffLimitPoints(shape, roi)));
            })
            .catch(() => {
                return [];
            })
            .finally(() => {
                pendingDecodesRef.current = Math.max(0, pendingDecodesRef.current - 1);
                if (pendingDecodesRef.current === 0) {
                    setIsDecoding(false);
                }
            });
    };

    const handlePointerUp = (event: PointerEvent<SVGSVGElement>) => {
        if (!ref.current) {
            return;
        }

        if (event.button !== 0 && event.button !== 2) {
            return;
        }

        if (previewShapes.length === 0) {
            return;
        }

        cancellableThrottledDecodingQueryFn.cancel();
        addAndSelectAnnotations(previewShapes, selectedLabel ? [selectedLabel] : []);
        setPreviewShapes([]);
        pendingDecodesRef.current = 0;
        setIsDecoding(false);
    };

    const handlePointerLeave = () => {
        cancellableThrottledDecodingQueryFn.cancel();
        setPreviewShapes([]);
        pendingDecodesRef.current = 0;
        setIsDecoding(false);
    };

    const previewAnnotations = previewShapes.map((shape, idx): Annotation => {
        return {
            shape,
            labels: selectedLabel ? [{ id: selectedLabel.id }] : [],
            id: `${idx}`,
        };
    });

    useEffect(() => {
        if (isError && !hasShownErrorToastRef.current) {
            toast({
                type: 'error',
                message: `
                Error in Segment Anything tool: ${error?.message ?? 'Unknown error, please try refreshing the page.'}`,
            });

            hasShownErrorToastRef.current = true;
        }

        if (!isError) {
            hasShownErrorToastRef.current = false;
        }
    }, [isError, error]);

    if (isLoading) {
        return <SAMLoading isLoading={isLoading} />;
    }

    return (
        <SvgToolCanvas
            ref={ref}
            image={image}
            canvasRef={canvasRef}
            aria-label='SAM tool canvas'
            onPointerMove={handleMouseMove}
            onPointerUp={handlePointerUp}
            onPointerLeave={handlePointerLeave}
            style={{
                cursor: isDecoding ? 'progress' : `url(${selectionCursor}) ${CURSOR_OFFSET}, auto`,
            }}
        >
            <PreviewAnnotations previewAnnotations={previewAnnotations} image={image} />
        </SvgToolCanvas>
    );
};
