// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import type { AnnotationDTO, Media } from '@/api/types';
import { ZoomProvider } from '@/components/zoom/zoom.provider';

import { AnnotationActionsProvider } from '../annotation-actions-provider.component';
import { AnnotationVisibilityProvider } from '../annotation-visibility-provider.component';
import { AnnotatorLabelsProvider } from '../annotator-labels-provider.component';
import type { AnnotatorMode } from '../annotator-mode';
import { AnnotatorProvider } from '../annotator-provider.component';
import { SelectAnnotationProvider } from '../select-annotation-provider.component';
import { CanvasSettingsProvider } from './primary-toolbar/settings/canvas-settings-provider.component';

type AnnotatorProvidersProps = {
    mediaItem: Media;
    initialAnnotationsDTO: AnnotationDTO[];
    initialPredictionsDTO: AnnotationDTO[];
    mode: AnnotatorMode;
    isReadOnly?: boolean;
    children: ReactNode;
};

export const AnnotatorProviders = ({
    mediaItem,
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
                            <AnnotationActionsProvider
                                mediaItem={mediaItem}
                                initialAnnotationsDTO={initialAnnotationsDTO}
                                initialPredictionsDTO={initialPredictionsDTO}
                                mode={mode}
                                isReadOnly={isReadOnly}
                            >
                                <SelectAnnotationProvider>{children}</SelectAnnotationProvider>
                            </AnnotationActionsProvider>
                        </AnnotatorLabelsProvider>
                    </CanvasSettingsProvider>
                </AnnotationVisibilityProvider>
            </ZoomProvider>
        </AnnotatorProvider>
    );
};
