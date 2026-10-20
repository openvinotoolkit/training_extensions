// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { renderHook } from 'test-utils/render';

import { http } from '../api/utils';
import { server } from '../msw-node-setup';
import { SUBSET_PARAM } from './use-dataset-filters-search-params.hook';
import * as datasetMediaFilterOptionsHook from './use-dataset-media-filter-options.hook';
import { useSelectAllDatasetMedia } from './use-select-all-dataset-media.hook';

const getQueryFromRequest = (request: Request) => new URL(request.url).searchParams;

describe('useSelectAllDatasetMedia', () => {
    afterEach(() => {
        rs.restoreAllMocks();
    });

    it('resolves the whole dataset in a single request, and reports which ids are images', async () => {
        let requestCount = 0;

        server.use(
            http.get('/api/projects/{project_id}/dataset/media/ids', () => {
                requestCount += 1;

                return HttpResponse.json({
                    items: [
                        { id: 'image-1', type: 'image' },
                        { id: 'video-1', type: 'video' },
                        { id: 'image-2', type: 'image' },
                    ],
                });
            })
        );

        const { result } = renderHook(() => useSelectAllDatasetMedia());

        act(() => {
            result.current.mutate();
        });

        await waitFor(() => {
            expect(result.current.isSuccess).toBe(true);
        });

        expect(requestCount).toBe(1);
        expect(result.current.data?.mediaIds).toEqual(['image-1', 'video-1', 'image-2']);
        expect(result.current.data?.imageIds).toEqual(['image-1', 'image-2']);
    });

    it('forwards the active dataset filters, without the sort parameters the endpoint rejects', async () => {
        let query: URLSearchParams | undefined;

        server.use(
            http.get('/api/projects/{project_id}/dataset/media/ids', ({ request }) => {
                query = getQueryFromRequest(request);

                return HttpResponse.json({ items: [{ id: 'image-1', type: 'image' }] });
            })
        );

        const { result } = renderHook(() => useSelectAllDatasetMedia(), {
            route: `/projects/123?${SUBSET_PARAM}=training`,
        });

        act(() => {
            result.current.mutate();
        });

        await waitFor(() => {
            expect(result.current.isSuccess).toBe(true);
        });

        expect(query?.getAll('subsets')).toEqual(['training']);
        expect(query?.has('sort_by')).toBe(false);
        expect(query?.has('sort_direction')).toBe(false);
    });

    it('surfaces a failed request as an error instead of a partial selection', async () => {
        server.use(
            http.get('/api/projects/{project_id}/dataset/media/ids', () => new HttpResponse(null, { status: 404 }))
        );

        const { result } = renderHook(() => useSelectAllDatasetMedia());

        act(() => {
            result.current.mutate();
        });

        await waitFor(() => {
            expect(result.current.isError).toBe(true);
        });

        expect(result.current.data).toBeUndefined();
    });

    it('discards the response when the filters change during the request', async () => {
        const filterOptions = rs
            .spyOn(datasetMediaFilterOptionsHook, 'useDatasetMediaFilterOptions')
            .mockReturnValue({ subsets: ['training'] });

        const { result, rerender } = renderHook(() => useSelectAllDatasetMedia());

        server.use(
            http.get('/api/projects/{project_id}/dataset/media/ids', () => {
                // Reached only once the request is in flight, so the hook has already captured
                // the filters it asked for. Switching subsets now makes the response outdated.
                filterOptions.mockReturnValue({ subsets: ['validation'] });
                rerender();

                return HttpResponse.json({ items: [{ id: 'image-1', type: 'image' }] });
            })
        );

        act(() => result.current.mutate());

        await waitFor(() => expect(result.current.isSuccess).toBe(true));

        expect(result.current.data).toBeNull();
    });
});
