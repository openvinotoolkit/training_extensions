// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { getMockedPrepareImportDatasetJob } from '../../../../../mocks/mock-job';
import { http } from '../../../../api/utils';
import { useImportDatasetToProject } from '../../../../hooks/storage/use-import-dataset-to-project.hook';
import { server } from '../../../../msw-node-setup';
import {
    ImportDatasetDialogStateProvider,
    useImportDatasetDialogState,
} from '../../providers/export-import-dataset-dialog-provider.component';
import { ImportDatasetToProject } from './Import-dataset-to-project.component';

vi.mock('../../../../api/tus-upload', async (importOriginal) => ({
    ...(await importOriginal<typeof import('../../../../api/tus-upload')>()),
    transferFile: vi.fn(async () => '00000000-0000-4000-8000-000000000001'),
}));

vi.mock('../../../../hooks/storage/use-import-dataset-to-project.hook');

describe('ImportDatasetToProject', () => {
    const mockedStagedDatasetId = 'staged-dataset-123';
    const mockedPrepareImportDatasetJob = getMockedPrepareImportDatasetJob({});

    const renderApp = (data: null | { id: string; fileName: string }, mockedAppendImportEntry = vi.fn()) => {
        vi.mocked(useImportDatasetToProject).mockReturnValue({
            getImportEntry: vi.fn().mockReturnValue(data),
            getAllImportEntries: vi.fn(),
            appendImportEntry: mockedAppendImportEntry,
            deleteImportEntry: vi.fn(),
            updateImportEntryStep: vi.fn(),
            updateImportEntry: vi.fn(),
        });

        server.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(
                    getMockedProject({
                        task: {
                            task_type: 'instance_segmentation',
                            exclusive_labels: false,
                        },
                    })
                );
            }),
            http.post('/api/staged_datasets:from-upload', () => {
                return HttpResponse.json(
                    {
                        id: mockedStagedDatasetId,
                        format: 'geti',
                        size: 123,
                        metadata: null,
                        compressed: true,
                        ready_for_export: false,
                        ready_for_import: true,
                    },
                    { status: 201 }
                );
            }),
            http.post('/api/jobs', () => {
                return HttpResponse.json(mockedPrepareImportDatasetJob, { status: 202 });
            })
        );

        const App = () => {
            const { datasetImportDialogState } = useImportDatasetDialogState();

            return (
                <>
                    <button onClick={datasetImportDialogState.open}>Open Import Dialog</button>
                    <ImportDatasetToProject />
                </>
            );
        };

        render(
            <ImportDatasetDialogStateProvider>
                <App />
            </ImportDatasetDialogStateProvider>
        );
    };

    it("appends import entry with 'preparing' step after file upload", async () => {
        const mockedAppendImportEntry = vi.fn();
        const file = new File(['file content'], 'test.zip', { type: 'application/zip' });

        renderApp(null, mockedAppendImportEntry);

        await userEvent.click(await screen.findByRole('button', { name: /open import dialog/i }));
        expect(await screen.findByText('Drop the dataset .zip file here')).toBeVisible();
        const uploadFileElement = screen.getByTestId(/upload-zip-file/i);

        await userEvent.upload(uploadFileElement, [file]);

        await waitFor(() => {
            expect(mockedAppendImportEntry).toHaveBeenCalledWith(
                expect.objectContaining({
                    step: 'preparing',
                    fileName: file.name,
                    stagedDatasetId: mockedStagedDatasetId,
                    prepareJobId: mockedPrepareImportDatasetJob.job_id,
                })
            );
        });
    });
});
