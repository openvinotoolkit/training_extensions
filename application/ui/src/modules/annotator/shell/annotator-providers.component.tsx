// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import type { AnnotationDTO } from '@/api/types';
import { ZoomProvider } from '@/components/zoom/zoom.provider';

import { AnnotationDocumentProvider } from '../annotation-document-provider.component';
import { AnnotationVisibilityProvider } from '../annotation-visibility-provider.component';
import { AnnotatorLabelsProvider } from '../annotator-labels-provider.component';
import type { AnnotatorMode } from '../annotator-mode';
import { AnnotatorProvider } from '../annotator-provider.component';
import { SelectAnnotationProvider } from '../select-annotation-provider.component';
import { CanvasSettingsProvider } from './primary-toolbar/settings/canvas-settings-provider.component';

type AnnotatorProvidersProps = {
    initialAnnotationsDTO: AnnotationDTO[];
    initialPredictionsDTO: AnnotationDTO[];
    mode: AnnotatorMode;
    isReadOnly?: boolean;
    children: ReactNode;
};

export const AnnotatorProviders = ({
    initialAnnotationsDTO,
    initialPredictionsDTO,
    mode,
    isReadOnly = false,
    children,
}: AnnotatorProvidersProps) => {
    return (
        <AnnotatorProvider>
            <ZoomProvider>
                <AnnotationVisibilityProvider>
                    <CanvasSettingsProvider>
                        <AnnotatorLabelsProvider>
                            <AnnotationDocumentProvider
                                initialAnnotationsDTO={initialAnnotationsDTO}
                                initialPredictionsDTO={initialPredictionsDTO}
                                mode={mode}
                                isReadOnly={isReadOnly}
                            >
                                <SelectAnnotationProvider>{children}</SelectAnnotationProvider>
                            </AnnotationDocumentProvider>
                        </AnnotatorLabelsProvider>
                    </CanvasSettingsProvider>
                </AnnotationVisibilityProvider>
            </ZoomProvider>
        </AnnotatorProvider>
    );
};
