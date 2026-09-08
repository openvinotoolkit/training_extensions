// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { useSearchParams } from 'react-router-dom';
import { render } from 'test-utils/render';

import { http } from '../../../../../../api/utils';
import { server } from '../../../../../../msw-node-setup';
import { UnassignMediaFromView } from './unassign-media-from-view.component';

const SearchParamsSpy = () => {
    const [searchParams] = useSearchParams();

    return <div data-testid={'search-params-spy'}>{searchParams.toString()}</div>;
};

const getSearchParams = () => new URLSearchParams(screen.getByTestId('search-params-spy').textContent ?? '');

const renderUnassignMediaFromView = ({
    selectedMediaIds = ['m1', 'm2'],
    resetSelectedMediaIds = vi.fn(),
    route = '/projects/123?datasetViewId=collection-one',
}: {
    selectedMediaIds?: string[];
    resetSelectedMediaIds?: () => void;
    route?: string;
} = {}) => {
    return render(
        <>
            <UnassignMediaFromView selectedMediaIds={selectedMediaIds} resetSelectedMediaIds={resetSelectedMediaIds} />
            <SearchParamsSpy />
        </>,
        {
            route,
            path: '/projects/:projectId',
        }
    );
};

describe('UnassignMediaFromView', () => {
    beforeEach(() => {
        server.use(
            http.delete(
                '/api/projects/{project_id}/dataset/views/{dataset_view_id}/media',
                () => new HttpResponse(null, { status: 204 })
            )
        );
    });

    it('displays unassign from inside a view', () => {
        renderUnassignMediaFromView();

        expect(getSearchParams().get('datasetViewId')).toBe('collection-one');
        expect(screen.getByRole('button', { name: 'Unassign from this view' })).toBeInTheDocument();
    });

    it('does not display unassign on the entire dataset', () => {
        renderUnassignMediaFromView({ route: '/projects/123' });

        expect(getSearchParams().get('datasetViewId')).toBeNull();
        expect(screen.queryByRole('button', { name: 'Unassign from this view' })).not.toBeInTheDocument();
    });

    it('does not display unassign without a selection', () => {
        renderUnassignMediaFromView({ selectedMediaIds: [] });

        expect(screen.queryByRole('button', { name: 'Unassign from this view' })).not.toBeInTheDocument();
    });

    it('removes the selected media from the view and clears the selection', async () => {
        let requestedId: string | undefined;
        let body: unknown;
        server.use(
            http.delete(
                '/api/projects/{project_id}/dataset/views/{dataset_view_id}/media',
                async ({ params, request }) => {
                    requestedId = params.dataset_view_id as string;
                    body = await request.json();
                    return new HttpResponse(null, { status: 204 });
                }
            )
        );

        const resetSelectedMediaIds = vi.fn();
        const user = userEvent.setup();
        renderUnassignMediaFromView({ selectedMediaIds: ['m1', 'm2'], resetSelectedMediaIds });

        await user.click(screen.getByRole('button', { name: 'Unassign from this view' }));

        await waitFor(() => expect(requestedId).toBe('collection-one'));
        expect(body).toEqual({ media_ids: ['m1', 'm2'] });
        expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument();
        expect(resetSelectedMediaIds).toHaveBeenCalled();
    });
});
