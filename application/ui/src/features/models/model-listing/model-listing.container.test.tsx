// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedDatasetRevision } from 'mocks/mock-dataset-revision';
import { getMockedModel, getMockedModelArchitecture } from 'mocks/mock-model';
import { getMockedPipeline } from 'mocks/mock-pipeline';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../api/utils';
import { paths } from '../../../constants/paths';
import { server } from '../../../msw-node-setup';
import { ModelListingContainer } from './model-listing.container';

const PROJECT_ID = '7b073838-99d3-42ff-9018-4e901eb047fc';

const renderModelListing = () => {
    return render(<ModelListingContainer />, {
        route: paths.project.models({ projectId: PROJECT_ID }),
        path: paths.project.models.pattern,
    });
};

describe('ModelListingContainer', () => {
    beforeEach(() => {
        server.use(
            http.get('/api/model_architectures', () => {
                const mockModelArchitectureBalance = getMockedModelArchitecture({
                    id: 'architecture-1',
                    name: 'YOLOX',
                });
                const mockModelArchitectureSpeed = getMockedModelArchitecture({
                    id: 'architecture-2',
                    name: 'Deim-DFine-L',
                });

                const mockModelArchitectureAccuracy = getMockedModelArchitecture({
                    id: 'architecture-3',
                    name: 'MobileNetV2',
                });

                return HttpResponse.json({
                    model_architectures: [
                        mockModelArchitectureBalance,
                        mockModelArchitectureSpeed,
                        mockModelArchitectureAccuracy,
                    ],
                    top_picks: {
                        balance: mockModelArchitectureBalance.id,
                        speed: mockModelArchitectureSpeed.id,
                        accuracy: mockModelArchitectureAccuracy.id,
                    },
                });
            }),
            http.get('/api/projects/{project_id}', () => HttpResponse.json(getMockedProject({ id: PROJECT_ID }))),
            http.get('/api/jobs', () => HttpResponse.json([])),
            http.get('/api/projects/{project_id}/dataset_revisions', () =>
                HttpResponse.json([getMockedDatasetRevision({ id: 'dataset-revision-1', name: 'Dataset Revision 1' })])
            ),
            http.get('/api/projects/{project_id}/pipeline', () =>
                HttpResponse.json(getMockedPipeline({ status: 'idle' }))
            ),
            http.get('/api/projects/{project_id}/dataset/views', () => HttpResponse.json([])),
            http.get('/api/system/devices/training', () => HttpResponse.json([{ type: 'cpu', name: 'CPU' }])),
            http.get('/api/model_architectures/{architecture_id}', () =>
                HttpResponse.json(getMockedModelArchitecture({ id: 'Object_Detection_YOLOX_X' }))
            )
        );
    });

    it('hiding failed models when only failed models exist does not show the "No models yet" screen', async () => {
        const user = userEvent.setup();

        server.use(
            http.get('/api/projects/{project_id}/models', () =>
                HttpResponse.json([
                    getMockedModel({ id: 'model-1', name: 'Failed YOLOX Model', training_info: { status: 'failed' } }),
                ])
            )
        );

        renderModelListing();

        expect(await screen.findByText('Failed YOLOX Model')).toBeInTheDocument();
        expect(screen.queryByText(/No models yet\./i)).not.toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'Model listing options' }));

        await user.click(screen.getByRole('menuitem', { name: /hide failed models/i }));

        expect(screen.queryByText(/No models yet\./i)).not.toBeInTheDocument();

        expect(screen.getByText('No models found')).toBeInTheDocument();

        expect(screen.getByRole('button', { name: 'Model listing options' })).toBeInTheDocument();
    });

    it('shows the "No models yet" screen when the project has no models at all', async () => {
        server.use(http.get('/api/projects/{project_id}/models', () => HttpResponse.json([])));

        renderModelListing();

        expect(await screen.findByText(/No models yet\./i)).toBeInTheDocument();
        expect(screen.getByText(/Train your first model to get started\./i)).toBeInTheDocument();
    });

    it('requests and renders model details only after the model row is expanded', async () => {
        const user = userEvent.setup();
        const model = getMockedModel({ id: 'model-1', name: 'YOLOX Model v1', architecture: 'architecture-1' });
        const requestedModelIds: string[] = [];

        server.use(
            http.get('/api/projects/{project_id}/models', () => HttpResponse.json([model])),
            http.get('/api/projects/{project_id}/models/{model_id}', ({ params }) => {
                requestedModelIds.push(String(params.model_id));

                return HttpResponse.json(model);
            })
        );

        renderModelListing();

        expect(await screen.findByText('YOLOX Model v1')).toBeInTheDocument();
        expect(screen.queryByRole('tablist', { name: 'Model details' })).not.toBeInTheDocument();
        expect(requestedModelIds).toEqual([]);

        await user.click(screen.getByText('YOLOX Model v1'));

        expect(await screen.findByRole('tablist', { name: 'Model details' })).toBeInTheDocument();
        expect(requestedModelIds).toEqual(['model-1']);
    });
});
