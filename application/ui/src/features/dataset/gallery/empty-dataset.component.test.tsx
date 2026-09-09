// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DATASET_VIEW_ID_PARAM } from 'hooks/use-dataset-view-id.hook';
import { useSearchParams } from 'react-router-dom';
import { render } from 'test-utils/render';

import { EmptyDataset } from './empty-dataset.component';

vi.mock('./toolbar/media-upload.component', () => ({
    MediaUpload: () => <button>Upload media</button>,
}));

vi.mock('../providers/export-import-dataset-dialog-provider.component', () => ({
    useImportDatasetDialogState: () => ({ datasetImportDialogState: { open: vi.fn() } }),
}));

const SearchParamsSpy = () => {
    const [searchParams] = useSearchParams();

    return <div data-testid={'search-params-spy'}>{searchParams.toString()}</div>;
};

const renderEmptyDataset = ({ hasActiveFilter, route }: { hasActiveFilter: boolean; route: string }) => {
    return render(
        <>
            <EmptyDataset hasActiveFilter={hasActiveFilter} />
            <SearchParamsSpy />
        </>,
        { route, path: '/projects/:projectId' }
    );
};

describe('EmptyDataset', () => {
    it('explains that the selected view is empty instead of the dataset', () => {
        renderEmptyDataset({
            hasActiveFilter: false,
            route: `/projects/123?${DATASET_VIEW_ID_PARAM}=collection-one`,
        });

        expect(screen.getByText(/This view has no media items/)).toBeInTheDocument();
        expect(screen.queryByText(/Your dataset is empty/)).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Import dataset' })).not.toBeInTheDocument();
    });

    it('returns to the entire dataset from an empty view', async () => {
        const user = userEvent.setup();
        renderEmptyDataset({
            hasActiveFilter: false,
            route: `/projects/123?${DATASET_VIEW_ID_PARAM}=collection-one`,
        });

        await user.click(screen.getByRole('button', { name: 'Go to Entire dataset' }));

        await waitFor(() => {
            expect(screen.getByTestId('search-params-spy')).not.toHaveTextContent(DATASET_VIEW_ID_PARAM);
        });
        expect(screen.getByText(/Your dataset is empty/)).toBeInTheDocument();
    });

    it('shows the filter message when a filter is active, even within a view', () => {
        renderEmptyDataset({
            hasActiveFilter: true,
            route: `/projects/123?${DATASET_VIEW_ID_PARAM}=collection-one`,
        });

        expect(screen.getByText(/No media items match your filter/)).toBeInTheDocument();
        expect(screen.queryByText(/This view has no media items/)).not.toBeInTheDocument();
    });
});
