// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act, waitFor } from '@testing-library/react';
import { getMockedDatasetItem } from 'mocks/mock-dataset-item';
import { getMockedMediaImage } from 'mocks/mock-media';
import { delay, HttpResponse } from 'msw';
import { renderHook } from 'test-utils/render';

import { http } from '../api/utils';
import { server } from '../msw-node-setup';
import { SORT_DIRECTION_PARAM } from './use-dataset-filters-search-params.hook';
import { useDatasetMediaWithReviewStatus } from './use-dataset-media-with-review-status.hook';
import { DATASET_VIEW_ID_PARAM } from './use-dataset-view-id.hook';

type HandlerOptions = {
    mediaTotal: number;
    datasetTotal: number;
    initialDelayMs?: number;
    mediaInitialDelayMs?: number;
    datasetInitialDelayMs?: number;
    mediaNextPageDelayMs?: number;
    datasetNextPageDelayMs?: number;
};

const getOffsetFromRequest = (request: Request) => {
    const url = new URL(request.url);

    return Number(url.searchParams.get('offset') ?? '0');
};

const getSortParamsFromRequest = (request: Request) => {
    const url = new URL(request.url);

    return {
        sortBy: url.searchParams.get('sort_by'),
        sortDirection: url.searchParams.get('sort_direction'),
    };
};

const applyDelayForOffset = async (offset: number, initialDelayMs: number, nextPageDelayMs: number) => {
    if (offset === 0 && initialDelayMs > 0) {
        await delay(initialDelayMs);
    }

    if (offset > 0 && nextPageDelayMs > 0) {
        await delay(nextPageDelayMs);
    }
};

const setupHandlers = ({
    mediaTotal,
    datasetTotal,
    initialDelayMs = 0,
    mediaInitialDelayMs,
    datasetInitialDelayMs,
    mediaNextPageDelayMs = 0,
    datasetNextPageDelayMs = 0,
}: HandlerOptions) => {
    const resolvedMediaInitialDelayMs = mediaInitialDelayMs ?? initialDelayMs;
    const resolvedDatasetInitialDelayMs = datasetInitialDelayMs ?? initialDelayMs;

    server.use(
        http.get('/api/projects/{project_id}/dataset/media', async ({ request }) => {
            const offset = getOffsetFromRequest(request);

            await applyDelayForOffset(offset, resolvedMediaInitialDelayMs, mediaNextPageDelayMs);

            return HttpResponse.json({
                items: [getMockedMediaImage({ id: `media-${offset + 1}` })],
                pagination: {
                    offset,
                    count: 1,
                    total: mediaTotal,
                    limit: 0,
                },
            });
        }),
        http.get('/api/projects/{project_id}/dataset/items', async ({ request }) => {
            const offset = getOffsetFromRequest(request);

            await applyDelayForOffset(offset, resolvedDatasetInitialDelayMs, datasetNextPageDelayMs);

            return HttpResponse.json({
                items: [getMockedDatasetItem({ id: `item-${offset + 1}`, user_reviewed: false })],
                pagination: {
                    offset,
                    count: 1,
                    total: datasetTotal,
                    limit: 0,
                },
            });
        })
    );
};

const getDatasetViewIdFromRequest = (request: Request) => {
    const url = new URL(request.url);

    return url.searchParams.get('dataset_view_id');
};

describe('useDatasetMediaWithReviewStatus', () => {
    describe('isPending', () => {
        it('stays true while either initial query is still pending so review-status badges do not pop in after thumbnails', async () => {
            setupHandlers({
                mediaTotal: 1,
                datasetTotal: 1,
                mediaInitialDelayMs: 0,
                datasetInitialDelayMs: 200,
            });

            const { result } = renderHook(() => useDatasetMediaWithReviewStatus());

            // Media items resolve first, but the review-status query is still
            // in flight — isPending must remain true to avoid rendering the
            // gallery without annotation-status badges.
            await waitFor(() => {
                expect(result.current.items.length).toBeGreaterThan(0);
            });

            expect(result.current.isPending).toBe(true);

            await waitFor(() => {
                expect(result.current.isPending).toBe(false);
            });
        });
    });

    describe('isFetchingNextPage', () => {
        it('is false during the initial load even while queries are pending', async () => {
            setupHandlers({ mediaTotal: 1, datasetTotal: 1, initialDelayMs: 100 });

            const { result } = renderHook(() => useDatasetMediaWithReviewStatus());

            // Initial load uses isPending, not isFetchingNextPage — otherwise
            // the gallery would render a tile-sized loader instead of the
            // full-page overlay.
            expect(result.current.isFetchingNextPage).toBe(false);

            await waitFor(() => {
                expect(result.current.isPending).toBe(false);
            });

            expect(result.current.isFetchingNextPage).toBe(false);
        });

        it('returns true when media items are fetching next page', async () => {
            setupHandlers({ mediaTotal: 40, datasetTotal: 1, mediaNextPageDelayMs: 100 });

            const { result } = renderHook(() => useDatasetMediaWithReviewStatus());

            await waitFor(() => {
                expect(result.current.isPending).toBe(false);
            });

            act(() => {
                result.current.fetchNextPage();
            });

            await waitFor(() => {
                expect(result.current.isFetchingNextPage).toBe(true);
            });

            await waitFor(() => {
                expect(result.current.isFetchingNextPage).toBe(false);
            });
        });

        it('returns true when dataset items are fetching next page', async () => {
            setupHandlers({ mediaTotal: 1, datasetTotal: 40, datasetNextPageDelayMs: 100 });

            const { result } = renderHook(() => useDatasetMediaWithReviewStatus());

            await waitFor(() => {
                expect(result.current.isPending).toBe(false);
            });

            act(() => {
                result.current.fetchNextPage();
            });

            await waitFor(() => {
                expect(result.current.isFetchingNextPage).toBe(true);
            });

            await waitFor(() => {
                expect(result.current.isFetchingNextPage).toBe(false);
            });
        });

        it('returns false when no requests are pending and no next page is being fetched', async () => {
            setupHandlers({ mediaTotal: 1, datasetTotal: 1 });

            const { result } = renderHook(() => useDatasetMediaWithReviewStatus());

            await waitFor(() => {
                expect(result.current.isPending).toBe(false);
            });

            expect(result.current.isFetchingNextPage).toBe(false);
        });
    });

    describe('sort direction', () => {
        const mockSortableResponses = () => {
            let mediaSortParams: { sortBy: string | null; sortDirection: string | null } | undefined;
            let datasetSortParams: { sortBy: string | null; sortDirection: string | null } | undefined;

            server.use(
                http.get('/api/projects/{project_id}/dataset/media', ({ request }) => {
                    mediaSortParams = getSortParamsFromRequest(request);

                    return HttpResponse.json({
                        items: [getMockedMediaImage({ id: 'media-1' })],
                        pagination: { offset: 0, count: 1, total: 1, limit: 0 },
                    });
                }),
                http.get('/api/projects/{project_id}/dataset/items', ({ request }) => {
                    datasetSortParams = getSortParamsFromRequest(request);

                    return HttpResponse.json({
                        items: [getMockedDatasetItem({ id: 'item-1', user_reviewed: false })],
                        pagination: { offset: 0, count: 1, total: 1, limit: 0 },
                    });
                })
            );

            return {
                getMediaSortParams: () => mediaSortParams,
                getDatasetSortParams: () => datasetSortParams,
            };
        };

        it('requests upload_date/creation_date sorted desc by default, for both media and dataset items', async () => {
            const { getMediaSortParams, getDatasetSortParams } = mockSortableResponses();

            const { result } = renderHook(() => useDatasetMediaWithReviewStatus());

            await waitFor(() => {
                expect(result.current.isPending).toBe(false);
            });

            expect(getMediaSortParams()).toEqual({ sortBy: 'upload_date', sortDirection: 'desc' });
            expect(getDatasetSortParams()).toEqual({ sortBy: 'creation_date', sortDirection: 'desc' });
        });

        it('requests ascending order for both media and dataset items when the URL sort param is "asc"', async () => {
            const { getMediaSortParams, getDatasetSortParams } = mockSortableResponses();

            const { result } = renderHook(() => useDatasetMediaWithReviewStatus(), {
                route: `/projects/123?${SORT_DIRECTION_PARAM}=asc`,
                path: '/projects/:projectId',
            });

            await waitFor(() => {
                expect(result.current.isPending).toBe(false);
            });

            expect(getMediaSortParams()).toEqual({ sortBy: 'upload_date', sortDirection: 'asc' });
            expect(getDatasetSortParams()).toEqual({ sortBy: 'creation_date', sortDirection: 'asc' });
        });
    });

    describe('dataset view', () => {
        const mockDatasetViewIdCapture = () => {
            let mediaDatasetViewId: string | null | undefined;
            let itemsDatasetViewId: string | null | undefined;

            server.use(
                http.get('/api/projects/{project_id}/dataset/media', ({ request }) => {
                    mediaDatasetViewId = getDatasetViewIdFromRequest(request);

                    return HttpResponse.json({
                        items: [getMockedMediaImage({ id: 'media-1' })],
                        pagination: { offset: 0, count: 1, total: 1, limit: 0 },
                    });
                }),
                http.get('/api/projects/{project_id}/dataset/items', ({ request }) => {
                    itemsDatasetViewId = getDatasetViewIdFromRequest(request);

                    return HttpResponse.json({
                        items: [getMockedDatasetItem({ id: 'item-1', user_reviewed: false })],
                        pagination: { offset: 0, count: 1, total: 1, limit: 0 },
                    });
                })
            );

            return {
                getMediaDatasetViewId: () => mediaDatasetViewId,
                getItemsDatasetViewId: () => itemsDatasetViewId,
            };
        };

        it('omits dataset_view_id from both requests when the entire dataset is selected', async () => {
            const { getMediaDatasetViewId, getItemsDatasetViewId } = mockDatasetViewIdCapture();

            const { result } = renderHook(() => useDatasetMediaWithReviewStatus());

            await waitFor(() => {
                expect(result.current.isPending).toBe(false);
            });

            expect(getMediaDatasetViewId()).toBeNull();
            expect(getItemsDatasetViewId()).toBeNull();
        });

        it('sends dataset_view_id on both requests when a view is selected', async () => {
            const { getMediaDatasetViewId, getItemsDatasetViewId } = mockDatasetViewIdCapture();

            const { result } = renderHook(() => useDatasetMediaWithReviewStatus(), {
                route: `/projects/123?${DATASET_VIEW_ID_PARAM}=collection-one`,
                path: '/projects/:projectId',
            });

            await waitFor(() => {
                expect(result.current.isPending).toBe(false);
            });

            expect(getMediaDatasetViewId()).toBe('collection-one');
            expect(getItemsDatasetViewId()).toBe('collection-one');
        });
    });
});
