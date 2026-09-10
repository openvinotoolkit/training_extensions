// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedJob } from 'mocks/mock-job';
import { getMockedLabel } from 'mocks/mock-labels';
import { getMockedProject } from 'mocks/mock-project';
import { getMockedStagedDataset } from 'mocks/mock-staged-dataset';
import { HttpResponse } from 'msw';

import { expect, http, test } from '../fixtures';
import { jobStatusStreamHandler } from './utils';

const STAGED_DATASET_ID = 'staged-dataset-456';

const exportJob = getMockedJob({ job_id: 'export-job-123', job_type: 'export_dataset' });

const mockedProject = getMockedProject({
    task: {
        task_type: 'detection',
        exclusive_labels: true,
        labels: [getMockedLabel({ name: 'red' }), getMockedLabel({ name: 'blue' })],
    },
});

const exportingJob = getMockedJob({
    job_id: exportJob.job_id,
    status: 'RUNNING',
    job_type: 'export_dataset',
    message: 'Exporting progress...',
    progress: 45,
    metadata: {
        dataset_id: STAGED_DATASET_ID,
        project_id: mockedProject.id,
        filters: {
            labels: mockedProject.task.labels?.map(({ name }) => name) ?? [],
            include_unannotated: true,
        },
    },
});

const doneJob = getMockedJob({
    ...exportingJob,
    metadata: { ...exportingJob.metadata, dataset_id: STAGED_DATASET_ID },
    message: 'Export completed',
    status: 'DONE',
    progress: 100,
});

test.describe('Export dataset', () => {
    test.beforeEach(({ network }) => {
        network.use(
            http.post('/api/jobs', async () => {
                return HttpResponse.json(exportJob, { status: 201 });
            }),
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(mockedProject, { status: 200 });
            }),
            http.get('/api/projects', () => {
                return HttpResponse.json([mockedProject]);
            }),
            http.get('/api/staged_datasets/{staged_dataset_id}', () => {
                return HttpResponse.json(getMockedStagedDataset({ id: STAGED_DATASET_ID }), { status: 200 });
            }),
            http.get('/api/staged_datasets/{staged_dataset_id}/zip', async () => {
                return HttpResponse.arrayBuffer(new ArrayBuffer(1024), {
                    headers: {
                        'content-type': 'application/zip',
                        'content-disposition': `attachment; filename="dataset_${STAGED_DATASET_ID}.zip"`,
                    },
                });
            }),
            http.get('/api/projects/{project_id}/dataset/items', () => {
                return HttpResponse.json({
                    items: [],
                    pagination: { total: 4, count: 4, limit: 10, offset: 0 },
                });
            })
        );
    });

    test('export dataset with default settings', async ({ network, page }) => {
        let deletedStagedDatasetId: string | undefined;

        network.use(
            http.get('/api/jobs/{job_id}', () => {
                return HttpResponse.json(exportingJob, { status: 200 });
            }),
            jobStatusStreamHandler({ [exportJob.job_id]: [exportingJob, doneJob] }),
            http.delete('/api/staged_datasets/{staged_dataset_id}', ({ params }) => {
                deletedStagedDatasetId = params.staged_dataset_id as string;
                return new HttpResponse(null, { status: 204 });
            })
        );

        await page.goto(`projects/${mockedProject.id}/dataset`);

        await page.getByRole('button', { name: 'import export dataset' }).click();
        await page.getByText('Export dataset').click();

        const dialog = page.getByRole('dialog');
        const projectLabels = mockedProject.task.labels ?? [];

        await test.step('Verify default export settings', async () => {
            await expect(dialog.getByRole('checkbox', { name: 'Select all items' })).toBeChecked();
            await expect(dialog.getByRole('row', { name: projectLabels[0]?.name })).toHaveAttribute(
                'aria-selected',
                'true'
            );
            await expect(dialog.getByRole('row', { name: projectLabels[1]?.name })).toHaveAttribute(
                'aria-selected',
                'true'
            );

            await expect(dialog.getByRole('checkbox', { name: 'Include media without annotations' })).toBeChecked();
        });

        await test.step('Submit export job', async () => {
            await dialog.getByRole('button', { name: 'Export' }).click();
        });

        await test.step('Verify export job progress', async () => {
            await expect(dialog).toBeHidden();
            await expect(page.getByText(/Labels:/)).toBeVisible();
            for (const label of projectLabels) {
                await expect(page.getByText(label.name)).toBeVisible();
            }
            await expect(page.getByText(String(exportingJob.message))).toBeVisible();
        });

        await test.step('Verify export job completes and download dataset', async () => {
            await page.waitForSelector(`text="Dataset is ready for download"`);

            const downloadButton = page.getByRole('button', { name: /download dataset/i });
            await expect(downloadButton).toBeVisible();
            await expect(downloadButton).toBeEnabled();

            const downloadPromise = page.waitForEvent('download');
            await downloadButton.click();
            const download = await downloadPromise;

            expect(download.suggestedFilename()).toBe(`dataset_${STAGED_DATASET_ID}.zip`);
        });

        await test.step('Close removes the staged dataset', async () => {
            const closeButton = page.getByRole('button', { name: /close export dataset status/i });
            await closeButton.click();

            await expect(page.getByText('Dataset is ready for download')).toBeHidden();
            await expect.poll(() => deletedStagedDatasetId).toBe(STAGED_DATASET_ID);
        });
    });

    test('cancel export job', async ({ network, page }) => {
        network.use(
            http.get('/api/jobs/{job_id}', () => {
                return HttpResponse.json(exportingJob, { status: 200 });
            }),
            http.post('/api/jobs/{job_id}:cancel', () => {
                return HttpResponse.json(getMockedJob({ ...exportingJob, status: 'CANCELLED' }), { status: 200 });
            })
        );

        await page.goto(`projects/${mockedProject.id}/dataset`);

        await page.getByRole('button', { name: 'import export dataset' }).click();
        await page.getByText('Export dataset').click();

        const dialog = page.getByRole('dialog');

        await test.step('Submit export job', async () => {
            await dialog.getByRole('button', { name: 'Export' }).click();
        });

        await test.step('Cancel the running export job', async () => {
            await expect(dialog).toBeHidden();
            await page.getByRole('button', { name: /cancel job dialog/i, exact: true }).click();

            const container = page.getByRole('alertdialog');
            await container.getByRole('button', { name: /Cancel Job/i, exact: true }).click();

            await expect(page.getByText(String(exportingJob.message))).toBeHidden();
        });
    });
});
