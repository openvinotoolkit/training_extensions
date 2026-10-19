// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { ExportDatasetJob } from '@/api/types';
import { screen } from '@testing-library/react';
import { userEvent } from '@testing-library/user-event';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { getMockedJobExportJob } from '../../../../../../../mocks/mock-job';
import { http } from '../../../../../../api/utils';
import { server } from '../../../../../../msw-node-setup';
import { ExportFailedJob } from './export-failed-job.component';

const mockedRemoveLsExportId = rs.fn();

rs.mock('../../../../../../hooks/storage/use-export-dataset.hook', () => ({
    useExportDataset: () => ({
        removeLsExportId: mockedRemoveLsExportId,
        lsExportIds: [],
        addLsExportId: rs.fn(),
    }),
}));

describe('ExportFailedJob', () => {
    const renderApp = (job: ExportDatasetJob) => {
        server.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(getMockedProject({ id: 'project-123' }));
            })
        );

        render(<ExportFailedJob job={job} />);
    };

    beforeEach(() => {
        rs.clearAllMocks();
    });

    it('removes from local storage when close button is clicked', async () => {
        const mockExportJob = getMockedJobExportJob({ status: 'FAILED' });
        renderApp(mockExportJob);

        const closeButton = await screen.findByRole('button', { name: 'close export dataset status' });
        await userEvent.click(closeButton);

        expect(mockedRemoveLsExportId).toHaveBeenCalledWith(mockExportJob.job_id);
    });

    it('displays error message and job message', async () => {
        const mockExportJob = getMockedJobExportJob({
            status: 'FAILED',
            message: 'Export failed due to validation error',
            error: 'Dataset validation failed: missing required fields',
        });
        renderApp(mockExportJob);

        expect(await screen.findByText('Export failed due to validation error')).toBeVisible();

        expect(await screen.findByText('Error: Dataset validation failed: missing required fields')).toBeVisible();
    });
});
