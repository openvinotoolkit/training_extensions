// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { ExportDatasetJob } from '@/api/types';
import { screen, waitFor } from '@testing-library/react';
import { userEvent } from '@testing-library/user-event';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { getMockedJobExportJob } from '../../../../../../../mocks/mock-job';
import { http } from '../../../../../../api/utils';
import { server } from '../../../../../../msw-node-setup';
import { downloadFile } from '../../../../../../platform/download-file';
import { ExportCompletedJob } from './export-completed-job.component';

vi.mock('../../../../../../platform/download-file', () => ({
    downloadFile: vi.fn(),
}));

const mockedRemoveLsExportId = vi.fn();

vi.mock('../../../../../../hooks/storage/use-export-dataset.hook', () => ({
    useExportDataset: () => ({
        removeLsExportId: mockedRemoveLsExportId,
        getLsExportIds: vi.fn(() => []),
        addLsExportId: vi.fn(),
    }),
}));

describe('ExportCompletedJob', () => {
    const renderApp = (job: ExportDatasetJob) => {
        server.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(getMockedProject({ id: 'project-123' }));
            }),
            http.get('/api/staged_datasets/{staged_dataset_id}', () => {
                return HttpResponse.json({});
            })
        );

        render(<ExportCompletedJob job={job} />);
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('downloads dataset when download button is clicked', async () => {
        const mockExportJob = getMockedJobExportJob({ status: 'FINISHED' });
        renderApp(mockExportJob);

        const downloadButton = await screen.findByRole('button', { name: 'download dataset' });
        await userEvent.click(downloadButton);

        expect(downloadFile).toHaveBeenCalledWith(
            expect.stringContaining(`/api/staged_datasets/${mockExportJob.metadata.dataset_id}/zip`),
            `dataset_${mockExportJob.metadata.dataset_id}.zip`,
            'Dataset download started'
        );
    });

    it('deletes staged dataset and removes from local storage when close button is clicked', async () => {
        const deleteStageFileSpy = vi.fn();

        server.use(
            http.delete('/api/staged_datasets/{staged_dataset_id}', () => {
                deleteStageFileSpy();
                return HttpResponse.json(null, { status: 204 });
            })
        );

        const mockExportJob = getMockedJobExportJob({ status: 'FINISHED' });
        renderApp(mockExportJob);

        const closeButton = await screen.findByRole('button', { name: 'close export dataset status' });
        await userEvent.click(closeButton);

        expect(deleteStageFileSpy).toHaveBeenCalled();
        expect(mockedRemoveLsExportId).toHaveBeenCalledWith(mockExportJob.job_id);
    });

    it('displays "Dataset is ready for download" message for a completed job', async () => {
        const mockExportJob = getMockedJobExportJob({ status: 'FINISHED' });
        renderApp(mockExportJob);

        expect(await screen.findByText('Dataset is ready for download')).toBeVisible();
    });

    it('shows job message when job is done with invalid staged dataset', async () => {
        const mockExportJob = getMockedJobExportJob({
            status: 'DONE',
            message: 'Export completed but dataset was cleaned up',
            metadata: { dataset_id: null, project_id: 'project-123', filters: { include_unannotated: false } },
        });
        renderApp(mockExportJob);

        expect(await screen.findByText('Export completed but dataset was cleaned up')).toBeVisible();
        const downloadButton = await screen.findByRole('button', { name: 'download dataset' });
        expect(downloadButton).toBeDisabled();
    });

    it('removes from local storage without delete request when dataset_id is nil', async () => {
        const deleteStageFileSpy = vi.fn();

        server.use(
            http.delete('/api/staged_datasets/{staged_dataset_id}', () => {
                deleteStageFileSpy();
                return HttpResponse.json(null, { status: 204 });
            })
        );

        const mockExportJob = getMockedJobExportJob({
            status: 'DONE',
            metadata: { dataset_id: null, project_id: 'project-123', filters: { include_unannotated: false } },
        });
        renderApp(mockExportJob);

        const closeButton = await screen.findByRole('button', { name: 'close export dataset status' });
        await userEvent.click(closeButton);

        await waitFor(() => {
            expect(mockedRemoveLsExportId).toHaveBeenCalledWith(mockExportJob.job_id);
        });
        expect(deleteStageFileSpy).not.toHaveBeenCalled();
    });

    it('shows toast notification when download is triggered', async () => {
        const mockExportJob = getMockedJobExportJob({ status: 'FINISHED' });
        renderApp(mockExportJob);

        const downloadButton = await screen.findByRole('button', { name: 'download dataset' });
        await userEvent.click(downloadButton);

        expect(downloadFile).toHaveBeenCalledWith(
            expect.stringContaining(`/api/staged_datasets/${mockExportJob.metadata.dataset_id}/zip`),
            `dataset_${mockExportJob.metadata.dataset_id}.zip`,
            'Dataset download started'
        );
    });
});
