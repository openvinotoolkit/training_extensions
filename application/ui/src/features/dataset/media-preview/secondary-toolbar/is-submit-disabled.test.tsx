// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { type ReactNode } from 'react';

import type { AnnotationDTO, Label } from '@/api/types';
import { waitFor } from '@testing-library/react';
import { getMockedShape } from 'mocks/mock-annotation';
import { getMockedLabel } from 'mocks/mock-labels';
import { getMockedMediaImage } from 'mocks/mock-media';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';

import { http } from '../../../../api/utils';
import { AnnotationDocumentProvider } from '../../../../modules/annotator/annotation-document-provider.component';
import type { AnnotatorMode } from '../../../../modules/annotator/annotator-mode';
import { server } from '../../../../msw-node-setup';
import { renderHook } from '../../../../test-utils/render';
import { useSubmitAnnotations } from '../api/use-submit-annotations';
import { getIsSubmitDisabled } from './is-submit-disabled';

type RenderIsSubmitDisabledParams = {
    mode?: AnnotatorMode;
    hasSubsetChanged?: boolean;
    isLoadingPredictions?: boolean;
    initialAnnotationsDTO?: AnnotationDTO[];
    initialPredictionsDTO?: AnnotationDTO[];
    labels?: Label[];
};

const renderIsSubmitDisabled = ({
    mode = 'annotation',
    hasSubsetChanged = false,
    isLoadingPredictions = false,
    initialAnnotationsDTO = [],
    initialPredictionsDTO = [],
    labels = [],
}: RenderIsSubmitDisabledParams) => {
    server.use(
        http.get('/api/projects/{project_id}', () =>
            HttpResponse.json(getMockedProject({ task: { task_type: 'detection', exclusive_labels: false, labels } }))
        )
    );

    const wrapper = ({ children }: { children: ReactNode }) => (
        <AnnotationDocumentProvider
            mode={mode}
            initialAnnotationsDTO={initialAnnotationsDTO}
            initialPredictionsDTO={initialPredictionsDTO}
        >
            {children}
        </AnnotationDocumentProvider>
    );

    return renderHook(
        () => {
            const submission = useSubmitAnnotations({ mediaItem: getMockedMediaImage(), hasSubsetChanged });

            return getIsSubmitDisabled({ isLoadingPredictions, ...submission });
        },
        { wrapper }
    );
};

describe('getIsSubmitDisabled', () => {
    const label1 = getMockedLabel({ id: 'label-1', name: 'Cat', color: '#FF0000' });

    it('annotation mode: disabled when annotations and subset are both unchanged', async () => {
        const annotationsDTO: AnnotationDTO[] = [
            { labels: [{ id: label1.id }], shape: getMockedShape({ type: 'rectangle' }), confidences: null },
        ];

        const { result } = renderIsSubmitDisabled({
            mode: 'annotation',
            labels: [label1],
            initialAnnotationsDTO: annotationsDTO,
            hasSubsetChanged: false,
        });

        await waitFor(() => expect(result.current).toBe(true));
    });

    it('annotation mode: enabled when annotations are unchanged but subset changed', async () => {
        const annotationsDTO: AnnotationDTO[] = [
            { labels: [{ id: label1.id }], shape: getMockedShape({ type: 'rectangle' }), confidences: null },
        ];

        const { result } = renderIsSubmitDisabled({
            mode: 'annotation',
            labels: [label1],
            initialAnnotationsDTO: annotationsDTO,
            hasSubsetChanged: true,
        });

        await waitFor(() => expect(result.current).toBe(false));
    });

    it('prediction mode: enabled when a prediction is present', async () => {
        const predictionsDTO: AnnotationDTO[] = [
            { labels: [{ id: label1.id }], shape: getMockedShape({ type: 'rectangle' }), confidences: [0.9] },
        ];

        const { result } = renderIsSubmitDisabled({
            mode: 'prediction',
            labels: [label1],
            initialPredictionsDTO: predictionsDTO,
        });

        await waitFor(() => expect(result.current).toBe(false));
    });

    it('prediction mode: disabled while loading predictions, even with a prediction present', async () => {
        const predictionsDTO: AnnotationDTO[] = [
            { labels: [{ id: label1.id }], shape: getMockedShape({ type: 'rectangle' }), confidences: [0.9] },
        ];

        const { result } = renderIsSubmitDisabled({
            mode: 'prediction',
            labels: [label1],
            initialPredictionsDTO: predictionsDTO,
            isLoadingPredictions: true,
        });

        await waitFor(() => expect(result.current).toBe(true));
    });
});
