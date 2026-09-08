// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedDatasetView } from 'mocks/mock-dataset-view';
import { HttpResponse } from 'msw';
import { useSearchParams } from 'react-router-dom';
import { render } from 'test-utils/render';

import { http } from '../../../../../../api/utils';
import { server } from '../../../../../../msw-node-setup';
import { DatasetView } from '../type';
import { SaveDatasetView } from './save-dataset-view.component';

const EXISTING_VIEW = getMockedDatasetView({ id: 'collection-one', name: 'Collection One' });

const SearchParamsSpy = () => {
    const [searchParams] = useSearchParams();

    return <div data-testid={'search-params-spy'}>{searchParams.toString()}</div>;
};

const getSearchParams = () => new URLSearchParams(screen.getByTestId('search-params-spy').textContent ?? '');

const renderSaveDatasetView = ({
    selectedMediaIds = ['m1', 'm2', 'm3'],
    resetSelectedMediaIds = vi.fn(),
    route = '/projects/123',
    datasetViews = [EXISTING_VIEW],
}: {
    selectedMediaIds?: string[];
    resetSelectedMediaIds?: () => void;
    route?: string;
    datasetViews?: DatasetView[];
} = {}) => {
    return render(
        <>
            <SaveDatasetView
                selectedMediaIds={selectedMediaIds}
                datasetViews={datasetViews}
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

describe('SaveDatasetView', () => {
    beforeEach(() => {
        server.use(
            http.post('/api/projects/{project_id}/dataset/views', () =>
                HttpResponse.json(getMockedDatasetView({ id: 'new-view-id', name: 'Cats' }))
            )
        );
    });

    it('displays save a view when media is selected on the entire dataset', () => {
        renderSaveDatasetView();

        expect(getSearchParams().get('datasetViewId')).toBeNull();
        expect(screen.getByRole('button', { name: 'Save view' })).toBeInTheDocument();
    });

    it('does not display save a view from inside another view', () => {
        renderSaveDatasetView({ route: '/projects/123?datasetViewId=collection-one' });

        expect(getSearchParams().get('datasetViewId')).toBe('collection-one');
        expect(screen.queryByRole('button', { name: 'Save view' })).not.toBeInTheDocument();
    });

    it('reports how many media items the new view will hold', async () => {
        const user = userEvent.setup();
        renderSaveDatasetView({ selectedMediaIds: ['m1', 'm2', 'm3'] });

        await user.click(screen.getByRole('button', { name: 'Save view' }));

        expect(await screen.findByText('Selected 3 media items')).toBeInTheDocument();
    });

    it('keeps save disabled for a blank name', async () => {
        const user = userEvent.setup();
        renderSaveDatasetView();

        await user.click(screen.getByRole('button', { name: 'Save view' }));
        expect(await screen.findByRole('button', { name: 'Save' })).toBeDisabled();

        await user.type(screen.getByLabelText('View name'), '   ');
        expect(screen.getByRole('button', { name: 'Save' })).toBeDisabled();
    });

    it('rejects a name that an existing view already uses', async () => {
        const user = userEvent.setup();
        renderSaveDatasetView();

        await user.click(screen.getByRole('button', { name: 'Save view' }));
        await user.type(await screen.findByLabelText('View name'), EXISTING_VIEW.name);

        expect(screen.getByText('A view with this name already exists.')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Save' })).toBeDisabled();
    });

    it('creates the view with the trimmed name and the selected media, then switches to it and clears the selection', async () => {
        let body: unknown;
        server.use(
            http.post('/api/projects/{project_id}/dataset/views', async ({ request }) => {
                body = await request.json();
                return HttpResponse.json(getMockedDatasetView({ id: 'new-view-id', name: 'Cats' }));
            })
        );

        const resetSelectedMediaIds = vi.fn();
        const user = userEvent.setup();
        renderSaveDatasetView({ selectedMediaIds: ['m1', 'm2', 'm3'], resetSelectedMediaIds });

        await user.click(screen.getByRole('button', { name: 'Save view' }));
        await user.type(await screen.findByLabelText('View name'), '  Cats  ');
        await user.click(screen.getByRole('button', { name: 'Save' }));

        await waitFor(() => expect(body).toEqual({ name: 'Cats', media_ids: ['m1', 'm2', 'm3'] }));
        await waitFor(() => expect(getSearchParams().get('datasetViewId')).toBe('new-view-id'));
        expect(resetSelectedMediaIds).toHaveBeenCalled();
        expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    });

    it('leaves the view and the selection untouched when the user closes', async () => {
        const resetSelectedMediaIds = vi.fn();
        const user = userEvent.setup();
        renderSaveDatasetView({ resetSelectedMediaIds });

        await user.click(screen.getByRole('button', { name: 'Save view' }));
        await user.click(await screen.findByRole('button', { name: 'Close' }));

        expect(getSearchParams().get('datasetViewId')).toBeNull();
        expect(resetSelectedMediaIds).not.toHaveBeenCalled();
    });
});
