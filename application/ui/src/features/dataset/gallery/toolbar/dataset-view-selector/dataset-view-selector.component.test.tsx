// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DATASET_VIEW_ID_PARAM } from 'hooks/use-dataset-view-id.hook';
import { getMockedDatasetView } from 'mocks/mock-dataset-view';
import { useSearchParams } from 'react-router-dom';
import { render } from 'test-utils/render';

import { DatasetViewSelector } from './dataset-view-selector.component';

const DATASET_VIEWS = [
    getMockedDatasetView({ id: 'collection-one', name: 'Collection One' }),
    getMockedDatasetView({ id: 'collection-two', name: 'Collection Two' }),
];

const SearchParamsSpy = () => {
    const [searchParams] = useSearchParams();

    return <div data-testid={'search-params-spy'}>{searchParams.toString()}</div>;
};

const renderDatasetViewSelector = ({
    route = '/projects/123',
    datasetViews = DATASET_VIEWS,
    resetSelectedMediaIds = vi.fn(),
}: {
    route?: string;
    datasetViews?: typeof DATASET_VIEWS;
    resetSelectedMediaIds?: () => void;
} = {}) => {
    return render(
        <>
            <DatasetViewSelector datasetViews={datasetViews} resetSelectedMediaIds={resetSelectedMediaIds} />
            <SearchParamsSpy />
        </>,
        {
            route,
            path: '/projects/:projectId',
        }
    );
};

const openDatasetViewSelector = async (user: ReturnType<typeof userEvent.setup>) => {
    await user.click(screen.getByRole('button', { name: 'Select dataset view' }));
};

describe('DatasetViewSelector', () => {
    it('shows "Entire dataset" selected when no datasetViewId param is present', () => {
        renderDatasetViewSelector();

        expect(screen.getByRole('button', { name: 'Select dataset view' })).toHaveTextContent('Entire dataset');
    });

    it('shows the matching view selected when the datasetViewId param is present', () => {
        renderDatasetViewSelector({ route: `/projects/123?${DATASET_VIEW_ID_PARAM}=collection-one` });

        expect(screen.getByRole('button', { name: 'Select dataset view' })).toHaveTextContent('Collection One');
    });

    it('adds the datasetViewId param when the user selects a view', async () => {
        const user = userEvent.setup();
        renderDatasetViewSelector();

        await openDatasetViewSelector(user);
        await user.click(screen.getByRole('listitem', { name: 'Collection One' }));

        expect(screen.getByTestId('search-params-spy')).toHaveTextContent(`${DATASET_VIEW_ID_PARAM}=collection-one`);
        expect(screen.getByRole('button', { name: 'Select dataset view' })).toHaveTextContent('Collection One');
    });

    it('removes the datasetViewId param when the user selects "Entire dataset"', async () => {
        const user = userEvent.setup();
        renderDatasetViewSelector({ route: `/projects/123?${DATASET_VIEW_ID_PARAM}=collection-one` });

        await openDatasetViewSelector(user);
        await user.click(screen.getByRole('listitem', { name: 'Entire dataset' }));

        expect(screen.getByTestId('search-params-spy')).not.toHaveTextContent(DATASET_VIEW_ID_PARAM);
        expect(screen.getByRole('button', { name: 'Select dataset view' })).toHaveTextContent('Entire dataset');
    });

    it('preserves other search params when selecting a view', async () => {
        const user = userEvent.setup();
        renderDatasetViewSelector({ route: '/projects/123?sortDirection=asc' });

        await openDatasetViewSelector(user);
        await user.click(screen.getByRole('listitem', { name: 'Collection One' }));

        expect(screen.getByTestId('search-params-spy')).toHaveTextContent('sortDirection=asc');
        expect(screen.getByTestId('search-params-spy')).toHaveTextContent(`${DATASET_VIEW_ID_PARAM}=collection-one`);
    });

    it('closes the popover after selecting a view', async () => {
        const user = userEvent.setup();
        renderDatasetViewSelector();

        await openDatasetViewSelector(user);
        expect(screen.getByRole('list', { name: 'Dataset views list' })).toBeInTheDocument();

        await user.click(screen.getByRole('listitem', { name: 'Collection One' }));

        await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    });

    it('closes the popover when the user selects the already selected view', async () => {
        const user = userEvent.setup();
        renderDatasetViewSelector({ route: `/projects/123?${DATASET_VIEW_ID_PARAM}=collection-one` });

        await openDatasetViewSelector(user);
        await user.click(screen.getByRole('listitem', { name: 'Collection One' }));

        await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
        expect(screen.getByTestId('search-params-spy')).toHaveTextContent(`${DATASET_VIEW_ID_PARAM}=collection-one`);
    });

    it('keeps the selected view when the user cancels the delete confirmation', async () => {
        const user = userEvent.setup();
        renderDatasetViewSelector({ route: `/projects/123?${DATASET_VIEW_ID_PARAM}=collection-one` });

        await openDatasetViewSelector(user);
        await user.click(screen.getByRole('button', { name: 'Dataset view actions for Collection One' }));
        await user.click(screen.getByRole('menuitem', { name: 'Delete' }));

        await user.click(await screen.findByRole('button', { name: 'Close' }));

        await waitFor(() => expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument());
        expect(screen.getByTestId('search-params-spy')).toHaveTextContent(`${DATASET_VIEW_ID_PARAM}=collection-one`);
        expect(screen.getByRole('button', { name: 'Select dataset view' })).toHaveTextContent('Collection One');
    });

    it('falls back to "Entire dataset" and strips an unknown datasetViewId param', async () => {
        renderDatasetViewSelector({ route: `/projects/123?${DATASET_VIEW_ID_PARAM}=unknown-view-id` });

        expect(screen.getByRole('button', { name: 'Select dataset view' })).toHaveTextContent('Entire dataset');
        await waitFor(() => {
            expect(screen.getByTestId('search-params-spy')).not.toHaveTextContent(DATASET_VIEW_ID_PARAM);
        });
    });

    it('cannot be opened when the project has no views', async () => {
        const user = userEvent.setup();
        renderDatasetViewSelector({ datasetViews: [] });

        const trigger = screen.getByRole('button', { name: 'Select dataset view' });
        expect(trigger).toHaveAttribute('aria-disabled', 'true');

        await user.click(trigger);

        expect(screen.queryByRole('list', { name: 'Dataset views list' })).not.toBeInTheDocument();
    });

    it('clears the media selection when the user switches view', async () => {
        const resetSelectedMediaIds = vi.fn();
        const user = userEvent.setup();
        renderDatasetViewSelector({ resetSelectedMediaIds });

        await openDatasetViewSelector(user);
        await user.click(screen.getByRole('listitem', { name: 'Collection One' }));

        expect(resetSelectedMediaIds).toHaveBeenCalledTimes(1);
    });

    it('keeps the media selection when the user re-picks the current view', async () => {
        const resetSelectedMediaIds = vi.fn();
        const user = userEvent.setup();
        renderDatasetViewSelector({
            route: `/projects/123?${DATASET_VIEW_ID_PARAM}=collection-one`,
            resetSelectedMediaIds,
        });

        await openDatasetViewSelector(user);
        await user.click(screen.getByRole('listitem', { name: 'Collection One' }));

        expect(resetSelectedMediaIds).not.toHaveBeenCalled();
    });
});
