// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { isEmpty } from 'lodash-es';

import { useAnnotations } from '../annotation-document-provider.component';
import { useAnnotationVisibility } from '../annotation-visibility-provider.component';
import { DEFAULT_ANNOTATION_STYLES } from '../utils';
import { AnnotationShapeRenderer } from './annotation-shape-renderer.component';
import { MaskAnnotations } from './mask-annotations.component';

type ReadOnlyAnnotationsProps = {
    width: number;
    height: number;
};

export const ReadOnlyAnnotations = ({ width, height }: ReadOnlyAnnotationsProps) => {
    const { annotations } = useAnnotations();
    const { isFocussed } = useAnnotationVisibility();

    return (
        <svg
            aria-label={'annotations'}
            data-testid={'annotation-layer'}
            width={width}
            height={height}
            tabIndex={-1}
            style={{
                position: 'absolute',
                inset: 0,
                outline: 'none',
                overflow: 'visible',
                ...DEFAULT_ANNOTATION_STYLES,
            }}
        >
            {!isEmpty(annotations) && (
                <MaskAnnotations annotations={annotations} width={width} height={height} isEnabled={isFocussed}>
                    {annotations.map((annotation) => (
                        <AnnotationShapeRenderer key={annotation.id} annotation={annotation} />
                    ))}
                </MaskAnnotations>
            )}
        </svg>
    );
};
