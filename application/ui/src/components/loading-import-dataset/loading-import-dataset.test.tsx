// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Job } from '@/api/types';
import { screen, waitFor } from '@testing-library/react';
import { userEvent } from '@testing-library/user-event';
import { getMockedPrepareImportDatasetJob } from 'mocks/mock-job';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { formatBytes } from '../../shared/util';
import { LoadingImportDataset } from './loading-import-dataset.component';

const stagedDatasetId = 'staged-dataset-123';

describe('LoadingImportDataset', () => {
    const renderApp = ({
        job,
        status = 200,
        fileName = 'test.zip',
        onSuccess,
        deleteEntry,
    }: {
        job: Job;
        status?: number;
        fileName?: string;
        onSuccess: () => void;
        deleteEntry: () => void;
    }) => {
        server.use(
            http.get('/api/jobs/{job_id}', () => HttpResponse.json(job, { status })),
            http.delete('/api/staged_datasets/{staged_dataset_id}', () => {
                return HttpResponse.json(null, { status: 204 });
            })
        );

        return render(
            <LoadingImportDataset
                size={0}
                jobId={job.job_id}
                fileName={fileName}
                onSuccess={onSuccess}
                deleteEntry={deleteEntry}
                stagedDatasetId={stagedDatasetId}
            />
        );
    };

    it('displays job message, error, and does not delete entry when job fails', async () => {
        const job = getMockedPrepareImportDatasetJob({
            status: 'FAILED',
            message: 'Import failed due to validation error',
            error: 'Dataset validation failed: missing required fields',
        });

        const mockedDeleteImportEntry = rs.fn();
        renderApp({ job, onSuccess: rs.fn(), deleteEntry: mockedDeleteImportEntry });

        expect(await screen.findByText('Import failed due to validation error')).toBeVisible();

        await userEvent.click(screen.getByText('Technical details'));

        expect(await screen.findByText('Dataset validation failed: missing required fields')).toBeVisible();
        expect(mockedDeleteImportEntry).not.toHaveBeenCalled();
    });

    it('displays file processing message and progress percentage when job is running', async () => {
        const fileName = 'dataset.zip';
        const job = getMockedPrepareImportDatasetJob({
            status: 'RUNNING',
            progress: 55.8,
            message: 'Preparing dataset for import',
        });

        renderApp({ job, fileName, onSuccess: rs.fn(), deleteEntry: rs.fn() });

        expect(await screen.findByText(`${fileName} file is being processed for import`)).toBeVisible();
        expect(await screen.findByText('56%')).toBeVisible();
    });

    it('calls onSuccess, deletes entry, and shows success message when job is done', async () => {
        const fileName = 'dataset.zip';
        const job = getMockedPrepareImportDatasetJob({ status: 'DONE' });

        const mockedOnSuccess = rs.fn();
        const mockedDeleteImportEntry = rs.fn();
        renderApp({ job, fileName, onSuccess: mockedOnSuccess, deleteEntry: mockedDeleteImportEntry });

        await waitFor(() => {
            expect(mockedOnSuccess).toHaveBeenCalled();
            expect(mockedDeleteImportEntry).toHaveBeenCalled();

            expect(screen.getByText(`Dataset ${fileName} ${formatBytes(0)} imported successfully.`)).toBeVisible();
        });
    });

    it('deletes entry without calling onSuccess when job query fails', async () => {
        const mockedOnSuccess = rs.fn();
        const mockedDeleteImportEntry = rs.fn();

        renderApp({
            job: { job_id: '123', detail: 'Job not found' } as unknown as Job,
            status: 404,
            onSuccess: mockedOnSuccess,
            deleteEntry: mockedDeleteImportEntry,
        });

        expect(await screen.findByText('An error occurred during import.')).toBeVisible();

        await userEvent.click(screen.getByText('Technical details'));

        expect(await screen.findByText('Job not found')).toBeVisible();
        expect(mockedDeleteImportEntry).toHaveBeenCalled();
        expect(mockedOnSuccess).not.toHaveBeenCalled();
    });
});
