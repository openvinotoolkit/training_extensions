// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { MouseEvent, ReactNode, useEffect, useRef } from 'react';

import { useHotkeys } from 'react-hotkeys-hook';

import { useAnnotationActions } from '../../../shared/annotator/annotation-actions-provider.component';
import { useLabelResolver } from '../../../shared/annotator/labels';
import { useSelectedAnnotations } from '../../../shared/annotator/select-annotation-provider.component';
import { HOTKEYS } from '../../../shared/hotkeys-definition';
import { useAnnotatorLabels } from '../annotator-labels-provider.component';
import { drawingStyles } from '../tools/polygon-tool/utils';
import { useAnnotation } from './annotation-context';

export const SelectableAnnotation = ({ children }: { children: ReactNode }) => {
    const annotation = useAnnotation();
    const { deleteAnnotations } = useAnnotationActions();
    const { setSelectedLabelId } = useAnnotatorLabels();
    const { setSelectedAnnotations, selectedAnnotations } = useSelectedAnnotations();
    const { resolveAnnotationLabel } = useLabelResolver();
    const elementRef = useRef<SVGGElement>(null);

    const isSelected = selectedAnnotations?.has(annotation.id);
    const selectionStyles = isSelected ? { stroke: 'var(--energy-blue-light)' } : {};

    // Focus the element when it becomes selected
    useEffect(() => {
        if (isSelected && elementRef.current) {
            elementRef.current.focus();
        }
    }, [isSelected]);

    const handleSelectAnnotation = (event: MouseEvent<SVGElement>) => {
        const annotationLabelId = annotation.labels[0]?.id;
        const hasShiftPressed = event.shiftKey;

        setSelectedLabelId(annotationLabelId ?? null);

        setSelectedAnnotations((selected) => {
            if (!hasShiftPressed) {
                return new Set([annotation.id]);
            }

            const newSelected = new Set(selected);

            if (newSelected.has(annotation.id)) {
                newSelected.delete(annotation.id);
            } else {
                newSelected.add(annotation.id);
            }

            return newSelected;
        });
    };

    const handleDeleteAnnotations = () => {
        if (selectedAnnotations.size === 0) {
            return;
        }

        const annotationsToDelete = Array.from(selectedAnnotations);

        setSelectedAnnotations(new Set());
        deleteAnnotations(annotationsToDelete);
    };

    useHotkeys(HOTKEYS.delete, () => {
        // Focus the parent SVG container to keep focus within the annotation area
        const parentSvg = elementRef.current?.closest('svg');
        if (parentSvg) {
            (parentSvg as SVGSVGElement).focus();
        }

        handleDeleteAnnotations();
    });

    useHotkeys(
        HOTKEYS.selectAllAnnotations,
        (event) => {
            event.preventDefault();

            setSelectedAnnotations((prev) => {
                return new Set([...prev, annotation.id]);
            });
        },
        [setSelectedAnnotations]
    );

    useHotkeys(
        HOTKEYS.deselectAllAnnotations,
        (event) => {
            event.preventDefault();

            setSelectedAnnotations(() => {
                return new Set();
            });
        },
        [setSelectedAnnotations]
    );

    return (
        <g
            ref={elementRef}
            aria-label={isSelected ? 'selected annotation' : 'annotation'}
            tabIndex={isSelected ? 0 : -1}
            onClick={handleSelectAnnotation}
            style={{
                ...drawingStyles(annotation.labels[0] ? (resolveAnnotationLabel(annotation.labels[0]) ?? null) : null),
                ...selectionStyles,
                zIndex: 999,
                outline: 'none',
                position: 'relative',
                pointerEvents: 'auto',
            }}
        >
            {children}
        </g>
    );
};
