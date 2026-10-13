// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import type { AnnotationDTO, Media } from '@/api/types';
import { ZoomProvider } from '@/components/zoom/zoom.provider';

import { AnnotationActionsProvider } from '../annotation-actions-provider.component';
import { AnnotationVisibilityProvider } from '../annotation-visibility-provider.component';
import { AnnotatorLabelsProvider } from '../annotator-labels-provider.component';
import type { AnnotatorMode } from '../annotator-mode';
import { CanvasSettingsProvider } from './primary-toolbar/settings/canvas-settings-provider.component';

type ReadOnlyAnnotatorProvidersProps = {
    mediaItem: Media;
    initialAnnotationsDTO: AnnotationDTO[];
    initialPredictionsDTO?: AnnotationDTO[];
    mode?: AnnotatorMode;
    children: ReactNode;
};

const EMPTY_PREDICTIONS_DTO: AnnotationDTO[] = [];

export const ReadOnlyAnnotatorProviders = ({
    mediaItem,
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
                        <AnnotationActionsProvider
                            mediaItem={mediaItem}
                            initialAnnotationsDTO={initialAnnotationsDTO}
                            initialPredictionsDTO={initialPredictionsDTO}
                            mode={mode}
                            isReadOnly
                        >
                            {children}
                        </AnnotationActionsProvider>
                    </AnnotatorLabelsProvider>
                </CanvasSettingsProvider>
            </AnnotationVisibilityProvider>
        </ZoomProvider>
    );
};
