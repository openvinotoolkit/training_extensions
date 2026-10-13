// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen, waitFor } from '@testing-library/react';
import { DATASET_VIEW_ID_PARAM } from 'hooks/use-dataset-view-id.hook';
import { getMockedDatasetItem } from 'mocks/mock-dataset-item';
import { getMockedDatasetView } from 'mocks/mock-dataset-view';
import { getMockedPipeline } from 'mocks/mock-pipeline';
import { getMockedProject } from 'mocks/mock-project';
import { getMockedTrainingConfiguration } from 'mocks/mock-training-configuration';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../api/utils';
import { paths } from '../../../constants/paths';
import { server } from '../../../msw-node-setup';
import { TrainModel } from './train-model.component';

describe('TrainModel', () => {
    beforeEach(() => {
        server.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(getMockedProject({ id: '123' }));
            }),
            http.get('/api/projects/{project_id}/pipeline', () => {
                return HttpResponse.json(getMockedPipeline({}));
            }),
            http.get('/api/projects/{project_id}/dataset_revisions', () => {
                return HttpResponse.json([]);
            }),
            http.get('/api/projects/{project_id}/dataset/views', () => {
                return HttpResponse.json([]);
            }),
            http.get('/api/projects/{project_id}/models', () => {
                return HttpResponse.json([]);
            }),
            http.get('/api/projects/{project_id}/models/{model_id}/training_configuration', () => {
                return HttpResponse.json({ parameters: getMockedTrainingConfiguration() });
            }),
            http.get('/api/projects/{project_id}/training_configuration', () => {
                return HttpResponse.json({ parameters: getMockedTrainingConfiguration() });
            }),
            http.get('/api/model_architectures', () => {
                return HttpResponse.json({
                    model_architectures: [],
                    top_picks: null,
                });
            }),
            http.get('/api/system/devices/training', () => {
                return HttpResponse.json([{ type: 'cpu', name: 'CPU' }]);
            })
        );
    });

    it('shows warning message when there are not enough annotated media items', async () => {
        server.use(
            http.get('/api/projects/{project_id}/dataset/items', () => {
                return HttpResponse.json({
                    items: [
                        getMockedDatasetItem({
                            id: '1',
                            subset: 'unassigned',
                        }),
                        getMockedDatasetItem({
                            id: '2',
                            subset: 'unassigned',
                        }),
                    ],
                    pagination: {
                        total: 2,
                        count: 2,
                        limit: 10,
                        offset: 0,
                    },
                });
            })
        );

        render(<TrainModel />);

        fireEvent.click(await screen.findByRole('button', { name: 'Train model' }));

        expect(
            await screen.findByText(/In order to train a model, you need to annotate at least 3 items/)
        ).toBeVisible();
    });

    it('does not show warning message when there are enough annotated media items', async () => {
        server.use(
            http.get('/api/projects/{project_id}/dataset/items', () => {
                return HttpResponse.json({
                    items: [
                        getMockedDatasetItem({
                            id: '1',
                            subset: 'unassigned',
                        }),
                        getMockedDatasetItem({
                            id: '2',
                            subset: 'unassigned',
                        }),
                        getMockedDatasetItem({
                            id: '3',
                            subset: 'unassigned',
                        }),
                        getMockedDatasetItem({
                            id: '4',
                            subset: 'unassigned',
                        }),
                    ],
                    pagination: {
                        total: 4,
                        count: 4,
                        limit: 10,
                        offset: 0,
                    },
                });
            })
        );

        render(<TrainModel />);

        fireEvent.click(await screen.findByRole('button', { name: 'Train model' }));

        await waitFor(() => {
            expect(
                screen.queryByText(/In order to train a model, you need to annotate at least 3 items/)
            ).not.toBeInTheDocument();
        });
    });

    describe('dataset selection', () => {
        beforeEach(() => {
            server.use(
                http.get('/api/projects/{project_id}/dataset/views', () => {
                    return HttpResponse.json([
                        getMockedDatasetView({ id: 'collection-one', name: 'Collection One' }),
                        getMockedDatasetView({ id: 'collection-two', name: 'Collection Two' }),
                    ]);
                }),
                http.get('/api/projects/{project_id}/dataset/items', () => {
                    return HttpResponse.json({
                        items: [],
                        pagination: { total: 0, count: 0, limit: 10, offset: 0 },
                    });
                })
            );
        });

        it('defaults to the current dataset and lists the available dataset views', async () => {
            render(<TrainModel />);

            fireEvent.click(await screen.findByRole('button', { name: 'Train model' }));

            const picker = await screen.findByTestId('select-dataset');
            expect(picker).toHaveTextContent('Use entire dataset');

            fireEvent.click(picker);

            expect(await screen.findByRole('option', { name: 'Collection One' })).toBeVisible();
            expect(screen.getByRole('option', { name: 'Collection Two' })).toBeVisible();
        });

        it('groups views and revisions under a heading, but not the current dataset entry', async () => {
            render(<TrainModel />);

            fireEvent.click(await screen.findByRole('button', { name: 'Train model' }));
            fireEvent.click(await screen.findByTestId('select-dataset'));

            expect(await screen.findByRole('group', { name: 'Dataset views' })).toBeVisible();
            expect(screen.getByRole('option', { name: 'Use entire dataset' })).toBeVisible();
            expect(screen.queryByRole('group', { name: 'Current dataset' })).not.toBeInTheDocument();
        });

        it('preselects the dataset view that is open on the Dataset screen', async () => {
            render(<TrainModel />, {
                route: `${paths.project.details({ projectId: '123' })}?${DATASET_VIEW_ID_PARAM}=collection-two`,
            });

            fireEvent.click(await screen.findByRole('button', { name: 'Train model' }));

            expect(await screen.findByTestId('select-dataset')).toHaveTextContent('Collection Two');
        });

        it('blocks training when the selected view has too few annotated items, even if the dataset has enough', async () => {
            server.use(
                http.get('/api/projects/{project_id}/dataset/items', ({ request }) => {
                    const datasetViewId = new URL(request.url).searchParams.get('dataset_view_id');
                    const total = datasetViewId === null ? 5 : 2;

                    return HttpResponse.json({
                        items: [],
                        pagination: { total, count: 0, limit: 10, offset: 0 },
                    });
                })
            );

            render(<TrainModel />);

            fireEvent.click(await screen.findByRole('button', { name: 'Train model' }));

            const picker = await screen.findByTestId('select-dataset');
            await waitFor(() => {
                expect(screen.queryByText(/you need to annotate at least 3 items/)).not.toBeInTheDocument();
            });

            fireEvent.click(picker);
            fireEvent.click(await screen.findByRole('option', { name: 'Collection One' }));

            expect(await screen.findByText(/at least 3 items in the selected dataset view/)).toBeVisible();
            expect(screen.getByRole('button', { name: 'Start' })).toBeDisabled();
        });
    });
});
