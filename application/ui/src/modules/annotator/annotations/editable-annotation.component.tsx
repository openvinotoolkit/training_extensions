// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, RefObject, useRef } from 'react';

import { useZoom } from '@/components/zoom/zoom.provider';
import { useEventListener } from 'hooks/event-listener.hook';

import { isLeftButton } from '../../../shared/buttons-utils';
import { useAnnotator } from '../annotator-provider.component';
import { useSelectedAnnotations } from '../select-annotation-provider.component';
import { useTool } from '../tool-provider.component';
import { EditBoundingBox } from '../tools/edit-bounding-box/edit-bounding-box.component';
import { EditPolygon } from '../tools/edit-polygon/edit-polygon.component';
import { useEditableAnnotationState } from '../use-editable-annotation-state.hook';
import { useAnnotation } from './annotation-context';
import { isPolygon, isRectangle } from './utils';

interface EditAnnotationProps {
    children: ReactNode;
}

const useUnselectAnnotationOnOutsideClick = (ref: RefObject<SVGGElement | null>) => {
    const { selectedAnnotations, setSelectedAnnotations } = useSelectedAnnotations();
    const { activeTool } = useTool();
    const { canvasRef } = useAnnotator();

    useEventListener(
        'pointerdown',
        (event) => {
            if (ref.current == null) {
                return;
            }

            if (event.ctrlKey || !isLeftButton(event)) {
                return;
            }

            if (activeTool === 'selection' && event.shiftKey) {
                return;
            }

            if (selectedAnnotations.size === 0) {
                return;
            }

            if (ref.current.contains(event.target as Node)) {
                return;
            }

            setSelectedAnnotations(new Set());
        },
        canvasRef
    );
};

const UnselectAnnotationOnOutsideClick = ({ ref }: { ref: RefObject<SVGGElement | null> }) => {
    useUnselectAnnotationOnOutsideClick(ref);

    return null;
};

export const EditableAnnotation = ({ children }: EditAnnotationProps) => {
    const { scale } = useZoom();
    const annotation = useAnnotation();
    const { isAnnotationEditable } = useEditableAnnotationState();
    const ref = useRef<SVGGElement>(null);

    const shouldDisplayEditAnchors = isAnnotationEditable(annotation.id);

    if (shouldDisplayEditAnchors) {
        if (isPolygon(annotation)) {
            return (
                <g ref={ref}>
                    <EditPolygon annotation={annotation} zoom={scale} />
                    <UnselectAnnotationOnOutsideClick ref={ref} />
                </g>
            );
        }

        if (isRectangle(annotation)) {
            const { shape } = annotation;

            return (
                <g ref={ref}>
                    <EditBoundingBox
                        key={`box-${shape.x}-${shape.y}-${shape.width}-${shape.height}`}
                        annotation={annotation}
                        zoom={scale}
                    />
                    <UnselectAnnotationOnOutsideClick ref={ref} />
                </g>
            );
        }
    }

    return <>{children}</>;
};
