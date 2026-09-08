// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { DatasetView, MediaImage, MediaWithPagination } from '@/api/types';
import { getMockedDatasetView } from 'mocks/mock-dataset-view';
import { getMockedMediaImage } from 'mocks/mock-media';
import { HttpResponse } from 'msw';

import { expect, http, test } from '../fixtures';

const PROJECT_ID = 'id-1';

const COLLECTION_ONE = getMockedDatasetView({ id: 'collection-one', name: 'Collection One', project_id: PROJECT_ID });
const COLLECTION_TWO = getMockedDatasetView({ id: 'collection-two', name: 'Collection Two', project_id: PROJECT_ID });

const IMAGE_A = getMockedMediaImage({ id: 'image-a', name: 'Image A' });
const IMAGE_B = getMockedMediaImage({ id: 'image-b', name: 'Image B' });
const IMAGE_C = getMockedMediaImage({ id: 'image-c', name: 'Image C' });
const ENTIRE_DATASET_MEDIA = [IMAGE_A, IMAGE_B, IMAGE_C];

const mediaResponse = (items: MediaImage[]) =>
    HttpResponse.json<MediaWithPagination>({
        items,
        pagination: { offset: 0, limit: 20, count: items.length, total: items.length },
    });

test.describe('Dataset views', () => {
    let views: DatasetView[];
    let viewMedia: Record<string, MediaImage[]>;

    test.beforeEach(({ network }) => {
        views = [COLLECTION_ONE, COLLECTION_TWO];
        viewMedia = {
            [COLLECTION_ONE.id]: [IMAGE_A],
            [COLLECTION_TWO.id]: [],
        };

        network.use(
            http.get('/api/projects/{project_id}/dataset/views', () => HttpResponse.json(views)),
            http.get('/api/projects/{project_id}/dataset/media', ({ query }) => {
                const viewId = query.get('dataset_view_id');

                return mediaResponse(viewId ? (viewMedia[viewId] ?? []) : ENTIRE_DATASET_MEDIA);
            })
        );
    });

    test.describe('Creating and switching views', () => {
        test('creates a view from the selected media and switches to it', async ({ datasetPage, network, page }) => {
            network.use(
                http.post('/api/projects/{project_id}/dataset/views', async ({ request }) => {
                    const body = (await request.json()) as { name: string; media_ids?: string[] | null };
                    const newView = getMockedDatasetView({ id: 'new-view', name: body.name, project_id: PROJECT_ID });

                    views = [...views, newView];
                    viewMedia[newView.id] = ENTIRE_DATASET_MEDIA.filter((item) =>
                        (body.media_ids ?? []).includes(item.id)
                    );

                    return HttpResponse.json(newView, { status: 201 });
                })
            );

            await datasetPage.goto();

            await datasetPage.selectMediaItem(IMAGE_A.id);
            await datasetPage.selectMediaItem(IMAGE_B.id);

            await datasetPage.views.saveViewAs('My view');

            await expect(datasetPage.views.getViewSelectorTrigger()).toHaveText('My view');
            await expect(page).toHaveURL(/datasetViewId=new-view/);
            await expect(datasetPage.getSelectedCountText(2)).toBeHidden();
            await expect(datasetPage.getMediaItemByName(IMAGE_A.name)).toBeVisible();
            await expect(datasetPage.getMediaItemByName(IMAGE_B.name)).toBeVisible();
            await expect(datasetPage.getMediaItemByName(IMAGE_C.name)).toBeHidden();
        });

        test('shows only the media that belongs to the selected view, then returns the whole dataset', async ({
            datasetPage,
            page,
        }) => {
            await datasetPage.goto();

            await expect(datasetPage.getImagesCountText(3)).toBeVisible();

            await datasetPage.views.selectView('Collection One');

            await expect(datasetPage.getImagesCountText(1)).toBeVisible();
            await expect(datasetPage.getMediaItemByName(IMAGE_A.name)).toBeVisible();
            await expect(datasetPage.getMediaItemByName(IMAGE_B.name)).toBeHidden();
            await expect(datasetPage.getMediaItemByName(IMAGE_C.name)).toBeHidden();

            await datasetPage.views.selectView('Entire dataset');

            await expect(datasetPage.getImagesCountText(3)).toBeVisible();
            await expect(page).not.toHaveURL(/datasetViewId/);
            await expect(datasetPage.getMediaItemByName(IMAGE_A.name)).toBeVisible();
            await expect(datasetPage.getMediaItemByName(IMAGE_B.name)).toBeVisible();
            await expect(datasetPage.getMediaItemByName(IMAGE_C.name)).toBeVisible();
        });

        test('keeps the selected view when navigating into the annotator and back', async ({
            datasetPage,
            network,
            page,
        }) => {
            network.use(
                http.get('/api/projects/{project_id}/dataset/media/{media_id}/binary', () => {
                    return new HttpResponse(null, { status: 200 });
                })
            );

            await datasetPage.goto('id-1', '?datasetViewId=collection-one');

            await expect(datasetPage.getMediaItemByName(IMAGE_A.name)).toBeVisible();

            await datasetPage.dblClickMediaItem(IMAGE_A.name);

            await page.goBack();

            await expect(datasetPage.views.getViewSelectorTrigger()).toHaveText('Collection One');
            await expect(datasetPage.getMediaItemByName(IMAGE_A.name)).toBeVisible();
            await expect(datasetPage.getMediaItemByName(IMAGE_B.name)).toBeHidden();
        });
    });

    test.describe('Assigning and unassigning media', () => {
        test('assigns media to an existing view and opens it from the success toast', async ({
            datasetPage,
            network,
            page,
        }) => {
            network.use(
                http.post('/api/projects/{project_id}/dataset/views/{dataset_view_id}/media', async ({ params }) => {
                    const assignedIds = [IMAGE_B.id, IMAGE_C.id];

                    viewMedia[params.dataset_view_id] = [
                        ...(viewMedia[params.dataset_view_id] ?? []),
                        ...ENTIRE_DATASET_MEDIA.filter((item) => assignedIds.includes(item.id)),
                    ];

                    return HttpResponse.json(null, { status: 204 });
                })
            );

            await datasetPage.goto();

            await datasetPage.selectMediaItem(IMAGE_B.id);
            await datasetPage.selectMediaItem(IMAGE_C.id);

            await datasetPage.views.assignToView('Collection Two');

            await expect(datasetPage.getImagesCountText(3)).toBeVisible();
            await expect(datasetPage.views.getViewSelectorTrigger()).toHaveText('Entire dataset');
            await expect(datasetPage.views.getOpenViewToastLink('Collection Two')).toBeVisible();
            await expect(datasetPage.getMediaItemByName(IMAGE_A.name)).toBeVisible();
            await expect(datasetPage.getMediaItemByName(IMAGE_B.name)).toBeVisible();
            await expect(datasetPage.getMediaItemByName(IMAGE_C.name)).toBeVisible();

            await datasetPage.views.getOpenViewToastLink('Collection Two').click();

            await expect(page).toHaveURL(/datasetViewId=collection-two/);
            await expect(datasetPage.views.getViewSelectorTrigger()).toHaveText('Collection Two');
            await expect(datasetPage.getImagesCountText(2)).toBeVisible();
            await expect(datasetPage.getMediaItemByName(IMAGE_B.name)).toBeVisible();
            await expect(datasetPage.getMediaItemByName(IMAGE_C.name)).toBeVisible();
            await expect(datasetPage.getMediaItemByName(IMAGE_A.name)).toBeHidden();
        });

        test('removes the selected media from the current view', async ({ datasetPage, network, page }) => {
            viewMedia[COLLECTION_ONE.id] = [IMAGE_A, IMAGE_B];

            network.use(
                http.delete('/api/projects/{project_id}/dataset/views/{dataset_view_id}/media', async ({ params }) => {
                    viewMedia[params.dataset_view_id] = viewMedia[params.dataset_view_id].filter(
                        (item) => item.id !== IMAGE_A.id
                    );

                    return HttpResponse.json(null, { status: 204 });
                })
            );

            await datasetPage.goto('id-1', '?datasetViewId=collection-one');

            await expect(datasetPage.getImagesCountText(2)).toBeVisible();
            await expect(datasetPage.getMediaItemByName(IMAGE_A.name)).toBeVisible();
            await expect(datasetPage.getMediaItemByName(IMAGE_B.name)).toBeVisible();

            await datasetPage.selectMediaItem(IMAGE_A.id);

            await expect(page.getByRole('dialog')).toBeHidden();
            await datasetPage.views.getUnassignButton().click();

            await expect(datasetPage.getImagesCountText(1)).toBeVisible();
            await expect(datasetPage.getSelectedCountText(1)).toBeHidden();
            await expect(datasetPage.getMediaItemByName(IMAGE_A.name)).toBeHidden();
            await expect(datasetPage.getMediaItemByName(IMAGE_B.name)).toBeVisible();
        });
    });

    test.describe('Renaming, deleting, and re-selecting views', () => {
        test('returns to the entire dataset after deleting the current view', async ({
            datasetPage,
            network,
            page,
        }) => {
            network.use(
                http.delete('/api/projects/{project_id}/dataset/views/{dataset_view_id}', async ({ params }) => {
                    views = views.filter((view) => view.id !== params.dataset_view_id);

                    return HttpResponse.json(null, { status: 204 });
                })
            );

            await datasetPage.goto('id-1', '?datasetViewId=collection-one');

            await datasetPage.views.deleteView('Collection One');
            await expect(datasetPage.views.getDeletedSuccessToast('Collection One')).toBeVisible();

            await expect(datasetPage.views.getViewSelectorTrigger()).toHaveText('Entire dataset');
            await expect(page).not.toHaveURL(/datasetViewId/);

            await datasetPage.views.openViewSelector();
            await expect(datasetPage.views.getViewRow('Collection One')).toBeHidden();
        });

        test('renames a view and shows the new name everywhere', async ({ datasetPage, network }) => {
            network.use(
                http.patch(
                    '/api/projects/{project_id}/dataset/views/{dataset_view_id}',
                    async ({ params, request }) => {
                        const body = (await request.json()) as { name: string };

                        views = views.map((view) =>
                            view.id === params.dataset_view_id ? { ...view, name: body.name } : view
                        );

                        return HttpResponse.json({ ...COLLECTION_ONE, name: body.name });
                    }
                )
            );

            await datasetPage.goto();

            await datasetPage.views.renameView('Collection One', 'Renamed');

            await datasetPage.views.openViewSelector();
            await expect(datasetPage.views.getViewRow('Renamed')).toBeVisible();
            await expect(datasetPage.views.getViewRow('Collection One')).toBeHidden();
        });
    });

    test.describe('View selector edge cases', () => {
        test('tells the user a view is empty instead of offering to upload', async ({ datasetPage, page }) => {
            viewMedia[COLLECTION_ONE.id] = [];

            await datasetPage.goto('id-1', '?datasetViewId=collection-one');

            await expect(datasetPage.views.getEmptyViewMessage()).toBeVisible();

            await datasetPage.views.getGoToEntireDatasetButton().click();

            await expect(datasetPage.getImagesCountText(3)).toBeVisible();
            await expect(page).not.toHaveURL(/datasetViewId/);
        });

        test('cannot be opened when the project has no views', async ({ datasetPage, network }) => {
            network.use(http.get('/api/projects/{project_id}/dataset/views', () => HttpResponse.json([])));

            await datasetPage.goto();

            await expect(datasetPage.views.getViewSelectorTrigger()).toHaveAttribute('aria-disabled', 'true');

            // eslint-disable-next-line playwright/no-force-option
            await datasetPage.views.getViewSelectorTrigger().click({ force: true });
            await expect(datasetPage.views.getViewsList()).toBeHidden();
        });

        test('falls back to the entire dataset for a link to a view that no longer exists', async ({
            datasetPage,
            page,
        }) => {
            await datasetPage.goto('id-1', '?datasetViewId=deleted-id');

            await expect(datasetPage.views.getViewSelectorTrigger()).toHaveText('Entire dataset');
            await expect(page).not.toHaveURL(/datasetViewId/);
        });

        test('clears the media selection when the user switches view', async ({ datasetPage }) => {
            await datasetPage.goto();

            await datasetPage.selectMediaItem(IMAGE_A.id);
            await datasetPage.selectMediaItem(IMAGE_B.id);

            await datasetPage.views.selectView('Collection One');

            await expect(datasetPage.getSelectedCountText(2)).toBeHidden();
            await expect(datasetPage.views.getSaveViewButton()).toBeHidden();
            await expect(datasetPage.views.getAssignButton()).toBeHidden();
            await expect(datasetPage.views.getUnassignButton()).toBeHidden();
        });

        test('keeps the selected view when the user clears all filters', async ({ datasetPage, page }) => {
            await datasetPage.goto('id-1', '?annotationStatusFilter=with_annotations&datasetViewId=collection-one');

            await expect(page.getByRole('button', { name: 'Clear all' })).toBeVisible();

            await page.getByRole('button', { name: 'Clear all' }).click();

            await expect(page.getByRole('button', { name: 'Clear all' })).toBeHidden();
            await expect(datasetPage.views.getViewSelectorTrigger()).toHaveText('Collection One');
            await expect(page).toHaveURL(/datasetViewId=collection-one/);
        });
    });
});
