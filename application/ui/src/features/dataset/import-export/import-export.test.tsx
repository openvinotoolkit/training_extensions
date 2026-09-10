// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen, waitFor } from '@testing-library/react';
import { getMockedDatasetStatistics } from 'mocks/mock-dataset-item';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { ImportDatasetDialogStateProvider } from '../providers/export-import-dataset-dialog-provider.component';
import { ImportExport } from './import-export.component';

describe('ImportExport', () => {
    it('opens the export dialog when export option is selected', async () => {
        let datasetStatisticsRequests = 0;

        server.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(getMockedProject({ id: '123' }));
            }),
            http.get('/api/projects/{project_id}/dataset/statistics', () => {
                datasetStatisticsRequests++;
                return HttpResponse.json(getMockedDatasetStatistics());
            }),
            http.get('/api/projects/{project_id}/dataset/items', () => {
                return HttpResponse.json({
                    pagination: {
                        total: 10,
                        offset: 0,
                        limit: 0,
                        count: 0,
                    },
                    items: [],
                });
            })
        );

        render(
            <ImportDatasetDialogStateProvider>
                <ImportExport />
            </ImportDatasetDialogStateProvider>
        );

        fireEvent.click(await screen.findByRole('button', { name: /import export dataset/i }));
        fireEvent.click(await screen.findByRole('menuitem', { name: /Export dataset/i }));

        expect(screen.getByRole('heading', { name: /Export settings/i })).toBeVisible();
        expect(datasetStatisticsRequests).toBe(0);

        fireEvent.click(screen.getByRole('radio', { name: 'COCO' }));
        await waitFor(() => expect(datasetStatisticsRequests).toBe(1));
    });
});
