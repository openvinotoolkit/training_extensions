// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act } from '@testing-library/react';
import { useSearchParams } from 'react-router-dom';

import { renderHook } from '../test-utils/render';
import { DATASET_VIEW_ID_PARAM, ENTIRE_DATASET_VIEW_ID, useDatasetViewId } from './use-dataset-view-id.hook';

describe('useDatasetViewId', () => {
    it('returns the entire dataset id when no param is present', () => {
        const { result } = renderHook(() => useDatasetViewId(), {
            route: '/projects/123',
            path: '/projects/:projectId',
        });

        const [datasetViewId] = result.current;

        expect(datasetViewId).toBe(ENTIRE_DATASET_VIEW_ID);
    });

    it('returns the entire dataset id when the param is an empty string', () => {
        const { result } = renderHook(() => useDatasetViewId(), {
            route: `/projects/123?${DATASET_VIEW_ID_PARAM}=`,
            path: '/projects/:projectId',
        });

        const [datasetViewId] = result.current;

        expect(datasetViewId).toBe(ENTIRE_DATASET_VIEW_ID);
    });

    it('returns the id from the search param', () => {
        const { result } = renderHook(() => useDatasetViewId(), {
            route: `/projects/123?${DATASET_VIEW_ID_PARAM}=collection-one`,
            path: '/projects/:projectId',
        });

        const [datasetViewId] = result.current;

        expect(datasetViewId).toBe('collection-one');
    });

    it('sets the id in the search params', () => {
        const { result } = renderHook(() => useDatasetViewId(), {
            route: '/projects/123',
            path: '/projects/:projectId',
        });

        act(() => {
            const [, setDatasetViewId] = result.current;
            setDatasetViewId('collection-one');
        });

        const [datasetViewId] = result.current;

        expect(datasetViewId).toBe('collection-one');
    });

    it('removes the param when setting the entire dataset id', () => {
        const { result } = renderHook(() => useDatasetViewId(), {
            route: `/projects/123?${DATASET_VIEW_ID_PARAM}=collection-one`,
            path: '/projects/:projectId',
        });

        expect(result.current[0]).toBe('collection-one');

        act(() => {
            const [, setDatasetViewId] = result.current;
            setDatasetViewId(ENTIRE_DATASET_VIEW_ID);
        });

        expect(result.current[0]).toBe(ENTIRE_DATASET_VIEW_ID);
    });

    it('preserves other search params when setting the id', () => {
        const useCombined = () => {
            const [searchParams] = useSearchParams();
            const [datasetViewId, setDatasetViewId] = useDatasetViewId();

            return { searchParams, datasetViewId, setDatasetViewId };
        };

        const { result } = renderHook(() => useCombined(), {
            route: '/projects/123?sortDirection=asc',
            path: '/projects/:projectId',
        });

        act(() => {
            result.current.setDatasetViewId('collection-one');
        });

        expect(result.current.datasetViewId).toBe('collection-one');
        expect(result.current.searchParams.get('sortDirection')).toBe('asc');
    });
});
