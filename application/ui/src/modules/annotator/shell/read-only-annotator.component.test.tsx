// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { AnnotationDTO } from '@/api/types';
import { fireEvent, screen } from '@testing-library/react';
import { getMockedAnnotation, getMockedShape } from 'mocks/mock-annotation';
import { getMockedLabel } from 'mocks/mock-labels';
import { getMockedMediaImage } from 'mocks/mock-media';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import type { AnnotatorMode } from '../annotator-mode';
import { ReadOnlyAnnotatorProviders } from './read-only-annotator-providers.component';
import { ReadOnlyAnnotator } from './read-only-annotator.component';

const label = getMockedLabel({ id: 'label-1' });
const mediaItem = getMockedMediaImage();
const image = new ImageData(new Uint8ClampedArray(4), 1, 1);

const annotationsDTO = [getMockedAnnotation({ labels: [{ id: label.id }] })];

const predictionsDTO = [
    getMockedAnnotation({
        shape: getMockedShape({ type: 'rectangle', x: 0, y: 0, width: 10, height: 10 }),
        labels: [{ id: label.id }],
    }),
];

const renderApp = ({
    mode = 'annotation' as AnnotatorMode,
    onModeChange,
    onSelectPreviousMediaItem,
    onSelectNextMediaItem,
    initialPredictionsDTO = predictionsDTO,
    initialAnnotationsDTO = annotationsDTO,
    isLoading = false,
}: {
    mode?: AnnotatorMode;
    onModeChange?: (mode: AnnotatorMode) => void;
    onSelectPreviousMediaItem?: () => void;
    onSelectNextMediaItem?: () => void;
    initialPredictionsDTO?: AnnotationDTO[];
    initialAnnotationsDTO?: AnnotationDTO[];
    isLoading?: boolean;
} = {}) => {
    return render(
        <ReadOnlyAnnotatorProviders
            initialAnnotationsDTO={initialAnnotationsDTO}
            initialPredictionsDTO={initialPredictionsDTO}
            mode={mode}
        >
            <ReadOnlyAnnotator
                mediaItem={mediaItem}
                image={image}
                subset={'training'}
                onClose={vi.fn()}
                mode={mode}
                onModeChange={onModeChange}
                isLoading={isLoading}
                onSelectPreviousMediaItem={onSelectPreviousMediaItem}
                onSelectNextMediaItem={onSelectNextMediaItem}
            />
        </ReadOnlyAnnotatorProviders>
    );
};

describe('ReadOnlyAnnotator', () => {
    beforeEach(() => {
        server.use(
            http.get('/api/projects/{project_id}', () =>
                HttpResponse.json(
                    getMockedProject({ task: { task_type: 'detection', exclusive_labels: true, labels: [label] } })
                )
            )
        );
    });

    it('does not render the annotation/prediction toggle when mode changes are not supported', async () => {
        renderApp();

        expect(await screen.findByRole('button', { name: 'Close' })).toBeInTheDocument();
        expect(screen.queryByTestId('annotator-modes-id')).not.toBeInTheDocument();
    });

    it('switches to prediction mode through the toggle', async () => {
        const onModeChange = vi.fn();

        renderApp({ onModeChange });

        fireEvent.click(await screen.findByRole('button', { name: 'Prediction' }));

        expect(onModeChange).toHaveBeenCalledWith('prediction');
    });

    it('hides the media navigation when there is no adjacent item', async () => {
        renderApp({ onModeChange: vi.fn(), onSelectNextMediaItem: undefined });

        expect(await screen.findByRole('button', { name: 'Close' })).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Next media item' })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Previous media item' })).not.toBeInTheDocument();
    });

    it('navigates to the next media item and disables the previous one', async () => {
        const onSelectNextMediaItem = vi.fn();

        renderApp({ onSelectNextMediaItem });

        expect(await screen.findByRole('button', { name: 'Previous media item' })).toBeDisabled();

        fireEvent.click(screen.getByRole('button', { name: 'Next media item' }));

        expect(onSelectNextMediaItem).toHaveBeenCalled();
    });

    it('shows a loading overlay and blocks navigation while the media item is loading', async () => {
        renderApp({
            mode: 'prediction',
            onModeChange: vi.fn(),
            onSelectPreviousMediaItem: vi.fn(),
            onSelectNextMediaItem: vi.fn(),
            isLoading: true,
        });

        expect(await screen.findByRole('progressbar', { name: 'Media canvas loading' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Previous media item' })).toBeDisabled();
        expect(screen.getByRole('button', { name: 'Next media item' })).toBeDisabled();
    });
});
