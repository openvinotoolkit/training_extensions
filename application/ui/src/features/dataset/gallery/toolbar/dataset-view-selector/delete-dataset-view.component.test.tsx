// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { DialogContainer } from '@geti-ui/ui';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedDatasetView } from 'mocks/mock-dataset-view';
import { delay, HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../../api/utils';
import { server } from '../../../../../msw-node-setup';
import { DeleteDatasetViewDialog } from './delete-dataset-view.component';
import { DatasetView } from './type';

const VIEW = getMockedDatasetView({ id: 'collection-one', name: 'Collection One' });

const renderDialog = ({
    onDismiss = vi.fn(),
    onSuccess = vi.fn(),
    datasetView = VIEW,
}: Partial<{ onSuccess: () => void; onDismiss: () => void; datasetView?: DatasetView }> = {}) => {
    return render(
        <DialogContainer onDismiss={onDismiss}>
            <DeleteDatasetViewDialog datasetView={datasetView} onSuccess={onSuccess} onCancel={vi.fn()} />
        </DialogContainer>
    );
};

describe('DeleteDatasetViewDialog', () => {
    beforeEach(() => {
        server.use(
            http.delete(
                '/api/projects/{project_id}/dataset/views/{dataset_view_id}',
                () => new HttpResponse(null, { status: 204 })
            ),
            http.get('/api/projects/{project_id}/dataset/views', () => HttpResponse.json([]))
        );
    });

    it('names the view in the confirmation', () => {
        renderDialog();

        expect(
            screen.getByText('Are you sure you want to delete the "Collection One" dataset view?')
        ).toBeInTheDocument();
    });

    it('deletes the view and reports success', async () => {
        let requestedId: string | undefined;
        server.use(
            http.delete('/api/projects/{project_id}/dataset/views/{dataset_view_id}', ({ params }) => {
                requestedId = params.dataset_view_id as string;
                return new HttpResponse(null, { status: 204 });
            })
        );

        const onSuccess = vi.fn();
        const user = userEvent.setup();
        renderDialog({ onSuccess });

        await user.click(screen.getByRole('button', { name: 'Delete' }));

        await waitFor(() => expect(onSuccess).toHaveBeenCalled());
        expect(requestedId).toBe(VIEW.id);
    });

    it('blocks a second delete while the first is in flight', async () => {
        server.use(
            http.delete('/api/projects/{project_id}/dataset/views/{dataset_view_id}', async () => {
                await delay('infinite');
                return new HttpResponse(null, { status: 204 });
            })
        );

        const user = userEvent.setup();
        renderDialog();

        await user.click(screen.getByRole('button', { name: 'Delete' }));

        await waitFor(() => expect(screen.getByRole('button', { name: 'Delete' })).toBeDisabled());
    });

    it('does not delete when the user closes the dialog', async () => {
        let requestFired = false;
        server.use(
            http.delete('/api/projects/{project_id}/dataset/views/{dataset_view_id}', () => {
                requestFired = true;
                return new HttpResponse(null, { status: 204 });
            })
        );

        const onDismiss = vi.fn();
        const user = userEvent.setup();
        renderDialog({ onDismiss });

        await user.click(screen.getByRole('button', { name: 'Close' }));

        expect(onDismiss).toHaveBeenCalled();
        expect(requestFired).toBe(false);
    });
});
