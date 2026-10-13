// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../../api/utils';
import { server } from '../../../../../msw-node-setup';
import type { DatasetGroup } from '../../types';
import { DatasetActions } from './dataset-actions.component';

const mockDataset: DatasetGroup = {
    id: 'dataset-123',
    name: 'Test Dataset',
    createdAt: '2025-01-10T00:00:00.000000+00:00',
    labelCount: 5,
    imageCount: 100,
    trainingSubsets: {
        training: 70,
        validation: 20,
        testing: 10,
    },
    filesDeleted: false,
};

describe('DatasetActions', () => {
    const renderApp = () => {
        server.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(getMockedProject({}));
            }),
            http.get('/api/projects/{project_id}/dataset_revisions/{dataset_revision_id}/items', ({ request }) => {
                const url = new URL(request.url);
                const annotationStatus = url.searchParams.get('annotation_status');

                if (annotationStatus === 'reviewed') {
                    return HttpResponse.json({
                        pagination: { total: 1, offset: 0, limit: 0, count: 0 },
                        items: [],
                    });
                }

                return HttpResponse.json({
                    pagination: { total: 1, offset: 0, limit: 0, count: 0 },
                    items: [],
                });
            })
        );

        render(<DatasetActions dataset={mockDataset} />);
    };

    it('should render menu with all items', async () => {
        renderApp();

        const menuButton = await screen.findByRole('button', { name: 'Dataset actions' });
        expect(menuButton).toBeVisible();

        await userEvent.click(menuButton);

        expect(screen.getByRole('menuitem', { name: 'Rename' })).toBeVisible();
        expect(screen.getByRole('menuitem', { name: 'Delete' })).toBeVisible();
        expect(screen.getByRole('menuitem', { name: 'Export' })).toBeVisible();
    });

    it('should open rename dialog when rename action is clicked', async () => {
        renderApp();

        const menuButton = await screen.findByRole('button', { name: 'Dataset actions' });
        await userEvent.click(menuButton);

        await userEvent.click(screen.getByRole('menuitem', { name: 'Rename' }));

        expect(screen.getByRole('dialog', { name: 'Rename dataset revision' })).toBeVisible();
        expect(screen.getByRole('textbox', { name: /Dataset revision name/ })).toHaveValue('Test Dataset');
    });

    it('should open delete dialog when delete action is clicked', async () => {
        renderApp();

        const menuButton = await screen.findByRole('button', { name: 'Dataset actions' });
        await userEvent.click(menuButton);

        await userEvent.click(screen.getByRole('menuitem', { name: 'Delete' }));

        expect(screen.getByRole('alertdialog', { name: 'Delete dataset revision' })).toBeVisible();
        expect(screen.getByText(/Are you sure you want to delete dataset revision/)).toBeVisible();
    });

    it('should open export dialog when export action is clicked', async () => {
        renderApp();

        const menuButton = await screen.findByRole('button', { name: 'Dataset actions' });
        await userEvent.click(menuButton);

        await userEvent.click(screen.getByRole('menuitem', { name: 'Export' }));

        expect(await screen.findByRole('heading', { name: 'Exported dataset statistics' })).toBeVisible();
    });
});
