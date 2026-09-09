// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import type { AnnotationDTO, Media } from '@/api/types';
import { ZoomProvider } from '@/components/zoom/zoom.provider';

import { AnnotationActionsProvider } from '../../../shared/annotator/annotation-actions-provider.component';
import { AnnotationVisibilityProvider } from '../../../shared/annotator/annotation-visibility-provider.component';
import type { AnnotatorMode } from '../../../shared/annotator/annotator-mode';
import { AnnotatorProvider } from '../../../shared/annotator/annotator-provider.component';
import { SelectAnnotationProvider } from '../../../shared/annotator/select-annotation-provider.component';
import { AnnotatorLabelsProvider } from '../../annotator/annotator-labels-provider.component';
import { CanvasSettingsProvider } from './primary-toolbar/settings/canvas-settings-provider.component';

type AnnotatorProvidersProps = {
    mediaItem: Media;
    initialAnnotationsDTO: AnnotationDTO[];
    initialPredictionsDTO: AnnotationDTO[];
    isUserReviewed: boolean;
    mode: AnnotatorMode;
    isReadOnly?: boolean;
    children: ReactNode;
};

export const AnnotatorProviders = ({
    mediaItem,
    initialAnnotationsDTO,
    initialPredictionsDTO,
    isUserReviewed,
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
                                isUserReviewed={isUserReviewed}
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
