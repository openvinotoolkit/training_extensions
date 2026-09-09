// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedDatasetView } from 'mocks/mock-dataset-view';
import { HttpResponse } from 'msw';
import { useSearchParams } from 'react-router-dom';
import { render } from 'test-utils/render';

import { http } from '../../../../../../api/utils';
import { server } from '../../../../../../msw-node-setup';
import { AssignToExistingView } from './assign-to-existing-view.component';

const COLLECTION_ONE = getMockedDatasetView({ id: 'collection-one', name: 'Collection One' });

const SearchParamsSpy = () => {
    const [searchParams] = useSearchParams();

    return <div data-testid={'search-params-spy'}>{searchParams.toString()}</div>;
};

const getSearchParams = () => new URLSearchParams(screen.getByTestId('search-params-spy').textContent ?? '');

const renderAssignToExistingView = ({
    datasetViews = [COLLECTION_ONE],
    selectedMediaIds = ['m1', 'm2'],
    resetSelectedMediaIds = vi.fn(),
    route = '/projects/123',
}: {
    datasetViews?: (typeof COLLECTION_ONE)[];
    selectedMediaIds?: string[];
    resetSelectedMediaIds?: () => void;
    route?: string;
} = {}) => {
    return render(
        <>
            <AssignToExistingView
                datasetViews={datasetViews}
                selectedMediaIds={selectedMediaIds}
                resetSelectedMediaIds={resetSelectedMediaIds}
            />
            <SearchParamsSpy />
        </>,
        {
            route,
            path: '/projects/:projectId',
        }
    );
};

const openDialog = async (user: ReturnType<typeof userEvent.setup>) => {
    await user.click(screen.getByRole('button', { name: 'Assign to existing view' }));
};

const pickView = async (user: ReturnType<typeof userEvent.setup>, name: string) => {
    await user.click(await screen.findByRole('button', { name: /assign to/i }));
    await user.click(await screen.findByRole('option', { name }));
};

describe('AssignToExistingView', () => {
    beforeEach(() => {
        server.use(
            http.post(
                '/api/projects/{project_id}/dataset/views/{dataset_view_id}/media',
                () => new HttpResponse(null, { status: 204 })
            )
        );
    });

    it('displays assign when media is selected on the entire dataset', () => {
        renderAssignToExistingView();

        expect(screen.getByRole('button', { name: 'Assign to existing view' })).toBeInTheDocument();
    });

    it('does not display assign from inside another view', () => {
        renderAssignToExistingView({ route: '/projects/123?datasetViewId=collection-one' });

        expect(screen.queryByRole('button', { name: 'Assign to existing view' })).not.toBeInTheDocument();
    });

    it('disables assigning when the project has no views', () => {
        renderAssignToExistingView({ datasetViews: [] });

        expect(screen.getByRole('button', { name: 'Assign to existing view' })).toBeDisabled();
    });

    it('keeps assign disabled until a view is picked', async () => {
        const user = userEvent.setup();
        renderAssignToExistingView();

        await openDialog(user);
        expect(await screen.findByRole('button', { name: 'Assign' })).toBeDisabled();

        await pickView(user, COLLECTION_ONE.name);

        expect(screen.getByRole('button', { name: 'Assign' })).toBeEnabled();
    });

    it('explains that other media in the view is unaffected', async () => {
        const user = userEvent.setup();
        renderAssignToExistingView();

        await openDialog(user);

        expect(
            await screen.findByText(
                'This operation will not affect other media that were already assigned to this view.'
            )
        ).toBeInTheDocument();
    });

    it('assigns the selected media to the chosen view, stays on the entire dataset, displays a link to the target view, and clears the selection', async () => {
        let requestedId: string | undefined;
        let body: unknown;
        server.use(
            http.post(
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
        renderAssignToExistingView({
            selectedMediaIds: ['m1', 'm2'],
            resetSelectedMediaIds,
            route: '/projects/123?sortBy=name',
        });

        await openDialog(user);
        await pickView(user, COLLECTION_ONE.name);
        await user.click(screen.getByRole('button', { name: 'Assign' }));

        await waitFor(() => expect(requestedId).toBe(COLLECTION_ONE.id));
        expect(body).toEqual({ media_ids: ['m1', 'm2'] });

        expect(getSearchParams().get('datasetViewId')).toBeNull();

        const toast = await screen.findByLabelText('toast');
        const link = within(toast).getByRole('link', { name: `Open ${COLLECTION_ONE.name} view` });

        expect(link).toHaveAttribute('href', expect.stringContaining('datasetViewId=collection-one'));
        expect(link).toHaveAttribute('href', expect.stringContaining('sortBy=name'));

        expect(resetSelectedMediaIds).toHaveBeenCalled();
    });
});
