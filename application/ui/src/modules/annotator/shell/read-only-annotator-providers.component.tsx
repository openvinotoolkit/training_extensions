// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import type { AnnotationDTO } from '@/api/types';
import { ZoomProvider } from '@/components/zoom/zoom.provider';

import { AnnotationDocumentProvider } from '../annotation-document-provider.component';
import { AnnotationVisibilityProvider } from '../annotation-visibility-provider.component';
import { AnnotatorLabelsProvider } from '../annotator-labels-provider.component';
import type { AnnotatorMode } from '../annotator-mode';
import { CanvasSettingsProvider } from './primary-toolbar/settings/canvas-settings-provider.component';

type ReadOnlyAnnotatorProvidersProps = {
    initialAnnotationsDTO: AnnotationDTO[];
    initialPredictionsDTO?: AnnotationDTO[];
    mode?: AnnotatorMode;
    children: ReactNode;
};

const EMPTY_PREDICTIONS_DTO: AnnotationDTO[] = [];

export const ReadOnlyAnnotatorProviders = ({
    initialAnnotationsDTO,
    initialPredictionsDTO = EMPTY_PREDICTIONS_DTO,
    mode = 'annotation',
    children,
}: ReadOnlyAnnotatorProvidersProps) => {
    return (
        <ZoomProvider>
            <AnnotationVisibilityProvider>
                <CanvasSettingsProvider>
                    <AnnotatorLabelsProvider>
                        <AnnotationDocumentProvider
                            initialAnnotationsDTO={initialAnnotationsDTO}
                            initialPredictionsDTO={initialPredictionsDTO}
                            mode={mode}
                            isReadOnly
                        >
                            {children}
                        </AnnotationDocumentProvider>
                    </AnnotatorLabelsProvider>
                </CanvasSettingsProvider>
            </AnnotationVisibilityProvider>
        </ZoomProvider>
    );
};
