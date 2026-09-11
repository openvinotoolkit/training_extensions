// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedDatasetRevision } from 'mocks/mock-dataset-revision';
import { getMockedModel, getMockedModelArchitecture } from 'mocks/mock-model';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';

import { expect, http, test } from '../fixtures';

const mockedModels = [
    getMockedModel({
        id: 'model-1',
        name: 'YOLOX Model v1',
        architecture: 'Object_Detection_YOLOX_X',
        training_info: {
            status: 'successful',
            label_schema_revision: { labels: [] },
            start_time: '2025-01-10T10:00:00.000000+00:00',
            end_time: '2025-01-10T12:30:00.000000+00:00',
            dataset_revision_id: 'dataset-1',
        },
    }),
    getMockedModel({
        id: 'model-2',
        name: 'YOLOX Model v2',
        architecture: 'Object_Detection_YOLOX_X',
        parent_revision: 'model-1',
        training_info: {
            status: 'successful',
            label_schema_revision: { labels: [] },
            start_time: '2025-01-11T10:00:00.000000+00:00',
            end_time: '2025-01-11T14:00:00.000000+00:00',
            dataset_revision_id: 'dataset-1',
        },
    }),
    getMockedModel({
        id: 'model-3',
        name: 'SSD Model',
        architecture: 'Object_Detection_SSD',
        training_info: {
            status: 'successful',
            label_schema_revision: { labels: [] },
            start_time: '2025-01-12T08:00:00.000000+00:00',
            end_time: '2025-01-12T10:00:00.000000+00:00',
            dataset_revision_id: 'dataset-2',
        },
    }),
];

const mockedModelArchitectures = [
    getMockedModelArchitecture({ id: 'Object_Detection_SSD', name: 'Object_Detection_SSD' }),
    getMockedModelArchitecture({ id: 'Object_Detection_YOLOX_X', name: 'Object_Detection_YOLOX_X' }),
    getMockedModelArchitecture({ id: 'Object_Detection_YOLOX_XS', name: 'Object_Detection_YOLOX_XS' }),
];

test.describe('Models', () => {
    test.beforeEach(({ network }) => {
        network.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(getMockedProject({ id: 'id-1' }));
            }),
            http.get('/api/projects/{project_id}/models', () => {
                return HttpResponse.json(mockedModels);
            }),
            http.get('/api/projects/{project_id}/dataset_revisions', () => {
                return HttpResponse.json([
                    getMockedDatasetRevision({ id: 'dataset-1', name: 'Dataset Revision 1' }),
                    getMockedDatasetRevision({ id: 'dataset-2', name: 'Dataset Revision 2' }),
                ]);
            }),
            http.get('/api/projects/{project_id}/models/{model_id}', ({ params }) => {
                const foundModel = mockedModels.find((model) => model.id === params.model_id);

                if (foundModel) {
                    return HttpResponse.json(getMockedModel(foundModel));
                }

                return new HttpResponse(null, { status: 404 });
            }),
            http.patch('/api/projects/{project_id}/models/{model_id}', async ({ request, params }) => {
                const body = (await request.json()) as { name: string };
                const foundModel = mockedModels.find((model) => model.id === params.model_id);

                if (foundModel) {
                    return HttpResponse.json({ ...foundModel, name: body.name });
                }

                return new HttpResponse(null, { status: 404 });
            }),
            http.delete('/api/projects/{project_id}/models/{model_id}', () => {
                return HttpResponse.json(null, { status: 204 });
            }),
            http.get('/api/model_architectures', () => {
                return HttpResponse.json({
                    model_architectures: mockedModelArchitectures,
                    top_picks: {
                        balance: mockedModelArchitectures[0].id,
                        speed: mockedModelArchitectures[1].id,
                        accuracy: mockedModelArchitectures[2].id,
                    },
                });
            }),
            http.get('/api/projects/{project_id}/dataset/items', () => {
                return HttpResponse.json({
                    items: [],
                    pagination: {
                        total: 0,
                        count: 0,
                        limit: 5,
                        offset: 0,
                    },
                });
            })
        );
    });

    test('displays models list', async ({ modelsPage }) => {
        await modelsPage.goto();

        await expect(modelsPage.getModelByName('YOLOX Model v1')).toBeVisible();
        await expect(modelsPage.getModelByName('YOLOX Model v2')).toBeVisible();
        await expect(modelsPage.getModelByName('SSD Model')).toBeVisible();
    });

    test('can expand model to show details tabs', async ({ modelsPage, page }) => {
        await modelsPage.goto();

        // Expand the first model
        await modelsPage.expandModel('YOLOX Model v1');

        const detailsTabs = page.getByRole('tablist', { name: 'Model details' });
        await expect(detailsTabs).toBeVisible();

        await expect(page.getByRole('tab', { name: 'Model variants' })).toBeVisible();
        await expect(page.getByRole('tab', { name: 'Model metrics' })).toBeVisible();
        await expect(page.getByRole('tab', { name: 'Training parameters' })).toBeVisible();
        await expect(page.getByRole('tab', { name: 'Training datasets' })).toBeVisible();
    });

    test('can search models by name', async ({ modelsPage }) => {
        await modelsPage.goto();

        await modelsPage.searchModels('SSD');

        await expect(modelsPage.getModelByName('SSD Model')).toBeVisible();
        await expect(modelsPage.getModelByName('YOLOX Model v1')).toBeHidden();
    });

    test('shows empty results when search has no matches', async ({ modelsPage, page }) => {
        await modelsPage.goto();

        await modelsPage.searchModels('NonExistentModel');

        await expect(page.getByRole('heading', { name: 'No models found' })).toBeVisible();
    });

    test('can change group by option', async ({ modelsPage, page }) => {
        await modelsPage.goto();

        await modelsPage.selectGroupBy('architecture');

        await expect(page.getByRole('heading', { name: 'Object_Detection_YOLOX_X', exact: true })).toBeVisible();
        await expect(page.getByRole('heading', { name: 'Object_Detection_SSD', exact: true })).toBeVisible();
    });

    test('can sort a group by clicking a column header', async ({ modelsPage }) => {
        await modelsPage.goto();

        await modelsPage.sortByColumn('dataset-1', 'Model Name');

        expect(await modelsPage.getModelNamesInOrder('dataset-1')).toEqual(['YOLOX Model v1', 'YOLOX Model v2']);

        // Clicking the same header again reverses the direction.
        await modelsPage.sortByColumn('dataset-1', 'Model Name');

        expect(await modelsPage.getModelNamesInOrder('dataset-1')).toEqual(['YOLOX Model v2', 'YOLOX Model v1']);
    });

    test('sorts each group independently', async ({ modelsPage }) => {
        await modelsPage.goto();

        await modelsPage.sortByColumn('dataset-1', 'Model Name');

        await expect(modelsPage.getColumnHeader('dataset-1', 'Model Name')).toHaveAccessibleName(
            'Model Name, sorted ascending'
        );
        await expect(modelsPage.getColumnHeader('dataset-2', 'Model Name')).toHaveAccessibleName('Sort by Model Name');
    });

    test('can toggle to show and hide failed models', async ({ modelsPage, network }) => {
        const failedModel = getMockedModel({
            id: 'model-3',
            name: 'Failed model',
            training_info: { status: 'failed' },
        });

        network.use(
            http.get('/api/projects/{project_id}/models', () => {
                return HttpResponse.json([...mockedModels, failedModel]);
            })
        );

        await modelsPage.goto();

        await expect(modelsPage.getModelByName('Failed model')).toBeVisible();

        await modelsPage.toggleShowHideFailedModels();

        await expect(modelsPage.getModelByName('Failed model')).toBeHidden();
    });

    test('can rename a model', async ({ modelsPage, network }) => {
        network.use(
            http.patch('/api/projects/{project_id}/models/{model_id}', async ({ request }) => {
                const body = (await request.json()) as { name: string };

                return HttpResponse.json(getMockedModel({ id: 'model-1', name: body.name }));
            }),
            http.get('/api/projects/{project_id}/models', () => {
                return HttpResponse.json([
                    getMockedModel({ id: 'model-1', name: 'Renamed Model' }),
                    ...mockedModels.slice(1),
                ]);
            })
        );

        await modelsPage.goto();

        await modelsPage.openModelMenu();
        await modelsPage.clickRenameAction();
        await modelsPage.renameModel('Renamed Model');

        await expect(modelsPage.getModelByName('Renamed Model')).toBeVisible();
    });

    test('can delete a model', async ({ modelsPage, network }) => {
        await modelsPage.goto();

        await expect(modelsPage.getModelByName('YOLOX Model v1')).toBeVisible();

        network.use(
            http.get('/api/projects/{project_id}/models', () => {
                return HttpResponse.json(mockedModels.slice(1));
            })
        );

        await modelsPage.openModelMenu();
        await modelsPage.clickDeleteAction();
        await modelsPage.confirmDeleteModel();

        await expect(modelsPage.getModelByName('YOLOX Model v1')).toBeHidden();
    });

    test('can delete model weights', async ({ modelsPage, network }) => {
        await modelsPage.goto();

        await expect(modelsPage.getModelByName('YOLOX Model v1')).toBeVisible();

        network.use(
            http.delete('/api/projects/{project_id}/models/{model_id}', () => {
                return HttpResponse.json(null, { status: 204 });
            }),
            http.get('/api/projects/{project_id}/models', () => {
                return HttpResponse.json([
                    getMockedModel({ ...mockedModels[0], files_deleted: true }),
                    ...mockedModels.slice(1),
                ]);
            })
        );

        await modelsPage.openModelMenu();
        await modelsPage.clickDeleteWeightsAction();
        await modelsPage.confirmDeleteWeights();

        // Model record is preserved
        await expect(modelsPage.getModelByName('YOLOX Model v1')).toBeVisible();

        // But variants are no longer available
        await modelsPage.expandModel('YOLOX Model v1');
        await expect(modelsPage.getModelDisclosure('model-1').getByText('No available model variants.')).toBeVisible();
    });

    test('can rename a dataset revision', async ({ modelsPage, network }) => {
        network.use(
            http.patch('/api/projects/{project_id}/dataset_revisions/{dataset_revision_id}', async ({ request }) => {
                const body = (await request.json()) as { name: string };

                return HttpResponse.json(getMockedDatasetRevision({ id: 'dataset-1', name: body.name }));
            }),
            http.get('/api/projects/{project_id}/dataset_revisions', () => {
                return HttpResponse.json([
                    getMockedDatasetRevision({ id: 'dataset-1', name: 'Renamed Dataset' }),
                    getMockedDatasetRevision({ id: 'dataset-2', name: 'Dataset Revision 2' }),
                ]);
            })
        );

        await modelsPage.goto();

        await modelsPage.openDatasetMenu();
        await modelsPage.clickRenameDatasetAction();
        await modelsPage.renameDatasetRevision('Renamed Dataset');

        await expect(modelsPage.getDatasetHeaderByName('Renamed Dataset')).toBeVisible();
    });

    test('can delete a dataset revision', async ({ modelsPage, network, page }) => {
        await modelsPage.goto();

        await expect(modelsPage.getThreeSectionRange('dataset-1')).toBeVisible();
        await expect(modelsPage.getThreeSectionRange('dataset-2')).toBeVisible();

        network.use(
            http.delete('/api/projects/{project_id}/dataset_revisions/{dataset_revision_id}', () => {
                return HttpResponse.json(null, { status: 204 });
            }),
            http.get('/api/projects/{project_id}/dataset_revisions', () => {
                return HttpResponse.json([
                    getMockedDatasetRevision({ id: 'dataset-1', name: 'Dataset Revision 1', files_deleted: true }),
                    getMockedDatasetRevision({ id: 'dataset-2', name: 'Dataset Revision 2' }),
                ]);
            })
        );

        await modelsPage.openDatasetMenu();
        await modelsPage.clickDeleteDatasetAction();
        await modelsPage.confirmDeleteDataset();

        await expect(modelsPage.getThreeSectionRange('dataset-1')).toBeHidden();
        await expect(modelsPage.getThreeSectionRange('dataset-2')).toBeVisible();

        await modelsPage.expandModel('YOLOX Model v1');
        await modelsPage.clickTrainingDatasetsTab();

        await expect(page.getByText('The files for this dataset revision have been deleted.')).toBeVisible();
        await expect(page.getByRole('heading', { name: /Training/ })).toBeHidden();
        await expect(page.getByRole('heading', { name: /Validation/ })).toBeHidden();
        await expect(page.getByRole('heading', { name: /Testing/ })).toBeHidden();
    });
});
