// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedDatasetView } from 'mocks/mock-dataset-view';
import { render } from 'test-utils/render';

import { DatasetViewItemsList } from './dataset-view-items-list.component';

const VIEW_A = getMockedDatasetView({ id: 'view-a', name: 'A' });
const VIEW_B = getMockedDatasetView({ id: 'view-b', name: 'B' });

describe('DatasetViewItemsList', () => {
    it('lists the entire dataset first, then the views in the order the API returned them', () => {
        render(
            <DatasetViewItemsList
                datasetViews={[VIEW_B, VIEW_A]}
                selectedDatasetViewId={null}
                onSelectDatasetView={vi.fn()}
                onOpenRenameDialog={vi.fn()}
                onOpenDeleteConfirmationDialog={vi.fn()}
            />
        );

        const names = screen.getAllByRole('listitem').map((item) => item.getAttribute('aria-label'));

        expect(names).toEqual(['Entire dataset', 'B', 'A']);
    });

    it('marks the entire dataset as selected and shows no actions for it when no view is chosen', () => {
        render(
            <DatasetViewItemsList
                datasetViews={[VIEW_A]}
                selectedDatasetViewId={null}
                onSelectDatasetView={vi.fn()}
                onOpenRenameDialog={vi.fn()}
                onOpenDeleteConfirmationDialog={vi.fn()}
            />
        );

        expect(screen.getByRole('listitem', { name: 'Entire dataset' })).toHaveAttribute('aria-current', 'true');
        expect(screen.getByRole('listitem', { name: 'A' })).not.toHaveAttribute('aria-current');
        expect(
            screen.queryByRole('button', { name: 'Dataset view actions for Entire dataset' })
        ).not.toBeInTheDocument();
    });

    it('marks the current view as selected', () => {
        render(
            <DatasetViewItemsList
                datasetViews={[VIEW_A, VIEW_B]}
                selectedDatasetViewId={VIEW_A.id}
                onSelectDatasetView={vi.fn()}
                onOpenRenameDialog={vi.fn()}
                onOpenDeleteConfirmationDialog={vi.fn()}
            />
        );

        expect(screen.getByRole('listitem', { name: 'A' })).toHaveAttribute('aria-current', 'true');
        expect(screen.getByRole('listitem', { name: 'B' })).not.toHaveAttribute('aria-current');
        expect(screen.getByRole('listitem', { name: 'Entire dataset' })).not.toHaveAttribute('aria-current');
    });

    it('selects a view or returns to the entire dataset when its row is clicked', async () => {
        const user = userEvent.setup();
        const onSelectDatasetView = vi.fn();

        render(
            <DatasetViewItemsList
                datasetViews={[VIEW_A]}
                selectedDatasetViewId={null}
                onSelectDatasetView={onSelectDatasetView}
                onOpenRenameDialog={vi.fn()}
                onOpenDeleteConfirmationDialog={vi.fn()}
            />
        );

        await user.click(screen.getByRole('listitem', { name: 'A' }));
        expect(onSelectDatasetView).toHaveBeenCalledWith(VIEW_A.id);

        await user.click(screen.getByRole('listitem', { name: 'Entire dataset' }));
        expect(onSelectDatasetView).toHaveBeenCalledWith(null);
    });

    it('opens rename or delete for a view without selecting it', async () => {
        const user = userEvent.setup();
        const onSelectDatasetView = vi.fn();
        const onOpenRenameDialog = vi.fn();
        const onOpenDeleteConfirmationDialog = vi.fn();

        render(
            <DatasetViewItemsList
                datasetViews={[VIEW_A]}
                selectedDatasetViewId={null}
                onSelectDatasetView={onSelectDatasetView}
                onOpenRenameDialog={onOpenRenameDialog}
                onOpenDeleteConfirmationDialog={onOpenDeleteConfirmationDialog}
            />
        );

        await user.click(screen.getByRole('button', { name: 'Dataset view actions for A' }));
        await user.click(await screen.findByRole('menuitem', { name: 'Rename' }));

        expect(onOpenRenameDialog).toHaveBeenCalledWith(VIEW_A);

        await user.click(screen.getByRole('button', { name: 'Dataset view actions for A' }));
        await user.click(await screen.findByRole('menuitem', { name: 'Delete' }));

        expect(onOpenDeleteConfirmationDialog).toHaveBeenCalledWith(VIEW_A);
        expect(onSelectDatasetView).not.toHaveBeenCalled();
    });
});
