// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedDatasetView } from 'mocks/mock-dataset-view';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../../api/utils';
import { server } from '../../../../../msw-node-setup';
import { RenameDatasetView } from './rename-dataset-view.component';

const CURRENT_VIEW = getMockedDatasetView({ id: 'collection-one', name: 'Collection One' });
const SIBLING_VIEW = getMockedDatasetView({ id: 'collection-two', name: 'Collection Two' });

describe('RenameDatasetView', () => {
    beforeEach(() => {
        server.use(
            http.patch('/api/projects/{project_id}/dataset/views/{dataset_view_id}', () =>
                HttpResponse.json(getMockedDatasetView({ name: 'Renamed' }))
            )
        );
    });

    it('pre-fills the field with the current name', () => {
        render(<RenameDatasetView datasetView={CURRENT_VIEW} datasetViews={[SIBLING_VIEW]} onClose={vi.fn()} />);

        const field = screen.getByLabelText('View name');

        expect(field).toHaveValue(CURRENT_VIEW.name);
        expect(field).toHaveFocus();
    });

    it('keeps save disabled until the name changes', async () => {
        const user = userEvent.setup();
        render(<RenameDatasetView datasetView={CURRENT_VIEW} datasetViews={[SIBLING_VIEW]} onClose={vi.fn()} />);

        expect(screen.getByRole('button', { name: 'Save' })).toBeDisabled();

        await user.type(screen.getByLabelText('View name'), ' Updated');

        expect(screen.getByRole('button', { name: 'Save' })).toBeEnabled();
    });

    it('keeps save disabled for a blank name', async () => {
        const user = userEvent.setup();
        render(<RenameDatasetView datasetView={CURRENT_VIEW} datasetViews={[SIBLING_VIEW]} onClose={vi.fn()} />);

        await user.clear(screen.getByLabelText('View name'));

        expect(screen.getByRole('button', { name: 'Save' })).toBeDisabled();
    });

    it('rejects a name that another view already uses', async () => {
        const user = userEvent.setup();
        render(<RenameDatasetView datasetView={CURRENT_VIEW} datasetViews={[SIBLING_VIEW]} onClose={vi.fn()} />);

        const field = screen.getByLabelText('View name');
        await user.clear(field);
        await user.type(field, SIBLING_VIEW.name);

        expect(field).toBeInvalid();
        expect(screen.getByText('A view with this name already exists.')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Save' })).toBeDisabled();
    });

    it('renames the view with the trimmed name and closes the dialog', async () => {
        let body: unknown;
        server.use(
            http.patch('/api/projects/{project_id}/dataset/views/{dataset_view_id}', async ({ request }) => {
                body = await request.json();
                return HttpResponse.json(getMockedDatasetView({ name: 'Renamed' }));
            })
        );

        const onClose = vi.fn();
        const user = userEvent.setup();
        render(<RenameDatasetView datasetView={CURRENT_VIEW} datasetViews={[SIBLING_VIEW]} onClose={onClose} />);

        const field = screen.getByLabelText('View name');
        await user.clear(field);
        await user.type(field, '  Renamed  ');
        await user.click(screen.getByRole('button', { name: 'Save' }));

        await waitFor(() => expect(body).toEqual({ name: 'Renamed' }));
        expect(onClose).toHaveBeenCalled();
    });

    it('closes without renaming when the user cancels', async () => {
        let requestFired = false;
        server.use(
            http.patch('/api/projects/{project_id}/dataset/views/{dataset_view_id}', () => {
                requestFired = true;
                return HttpResponse.json(getMockedDatasetView({ name: 'Renamed' }));
            })
        );

        const onClose = vi.fn();
        const user = userEvent.setup();
        render(<RenameDatasetView datasetView={CURRENT_VIEW} datasetViews={[SIBLING_VIEW]} onClose={onClose} />);

        await user.click(screen.getByRole('button', { name: 'Cancel' }));

        expect(onClose).toHaveBeenCalled();
        expect(requestFired).toBe(false);
    });
});
