// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { uploadDatasetArchive } from '@/api';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedStagedDataset } from 'mocks/mock-staged-dataset';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { getMockedPrepareImportDatasetJob } from '../../../mocks/mock-job';
import { http } from '../../api/utils';
import { server } from '../../msw-node-setup';
import { ImportUploadFile } from './import-upload-file.component';

vi.mock('@/api', async (importOriginal) => ({
    ...(await importOriginal<typeof import('@/api')>()),
    uploadDatasetArchive: vi.fn(),
}));

describe('ImportUploadFile', () => {
    const validFile = new File(['file content'], 'test.zip', { type: 'application/zip' });
    const inValidFiles = new File(['foo'], 'video.mov', { type: 'video/quicktime' });
    const mockedStagedDatasetId = 'staged-dataset-123';
    const mockedPrepareImportDatasetJob = getMockedPrepareImportDatasetJob({});

    const renderApp = () => {
        vi.mocked(uploadDatasetArchive).mockResolvedValue(
            getMockedStagedDataset({ id: mockedStagedDatasetId, size: 123 })
        );
        server.use(
            http.post('/api/jobs', () => {
                return HttpResponse.json(mockedPrepareImportDatasetJob, { status: 202 });
            })
        );
        const mockedOnFileUploaded = vi.fn();

        render(<ImportUploadFile formatOptions='coco' onFileUploaded={mockedOnFileUploaded} />);

        return mockedOnFileUploaded;
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('invalid file extension', async () => {
        const mockedOnFileUploaded = renderApp();

        const uploadFileElement = screen.getByTestId(/upload-zip-file/i);

        await userEvent.upload(uploadFileElement, [inValidFiles]);

        expect(screen.getByText(/Unsupported file format. Please upload a valid .zip file./i)).toBeVisible();

        await waitFor(() => {
            expect(mockedOnFileUploaded).not.toHaveBeenCalled();
        });
    });

    it('valid file extension', async () => {
        const mockedOnFileUploaded = renderApp();

        const uploadFileElement = screen.getByTestId(/upload-zip-file/i);

        await userEvent.upload(uploadFileElement, [validFile]);

        await waitFor(() => {
            expect(mockedOnFileUploaded).toHaveBeenCalledWith(
                expect.objectContaining({
                    stagedDatasetId: mockedStagedDatasetId,
                    prepareJobId: mockedPrepareImportDatasetJob.job_id,
                })
            );
        });
    });

    it('shows the transfer progress before processing the staged archive', async () => {
        renderApp();
        let finishStaging: ((result: ReturnType<typeof getMockedStagedDataset>) => void) | undefined;
        vi.mocked(uploadDatasetArchive).mockImplementation(
            async (_file, options) =>
                new Promise((resolve) => {
                    options?.onProgress?.(3);
                    finishStaging = resolve;
                })
        );

        await userEvent.upload(screen.getByTestId(/upload-zip-file/i), [validFile]);
        expect(await screen.findByText('3 B of 12 B')).toBeVisible();
        expect(screen.getByRole('progressbar', { name: 'Upload progress' })).toHaveAttribute('aria-valuenow', '25');

        finishStaging?.(getMockedStagedDataset({ id: mockedStagedDatasetId, size: 123 }));
        await waitFor(() => expect(screen.queryByText('3 B of 12 B')).not.toBeInTheDocument());
    });

    it('cancels the transfer without preparing the import or reporting an error', async () => {
        const mockedOnFileUploaded = renderApp();
        vi.mocked(uploadDatasetArchive).mockImplementation(
            async (_file, options) =>
                new Promise((_resolve, reject) => {
                    options?.onProgress?.(3);
                    options?.signal?.addEventListener('abort', () =>
                        reject(new DOMException('The upload was cancelled.', 'AbortError'))
                    );
                })
        );

        await userEvent.upload(screen.getByTestId(/upload-zip-file/i), [validFile]);
        await userEvent.click(await screen.findByRole('button', { name: 'Cancel' }));

        expect(await screen.findByText('Drop the dataset .zip file here')).toBeVisible();
        expect(vi.mocked(uploadDatasetArchive).mock.calls[0][1]?.signal?.aborted).toBe(true);
        expect(screen.queryByText(/cancelled/i)).not.toBeInTheDocument();
        expect(mockedOnFileUploaded).not.toHaveBeenCalled();
    });
});
