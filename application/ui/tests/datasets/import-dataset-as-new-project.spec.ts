// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { type Page } from '@playwright/test';
import { getMockedJob } from 'mocks/mock-job';
import { getMockedStagedDataset } from 'mocks/mock-staged-dataset';
import { HttpResponse } from 'msw';

import { expect, http, test } from '../fixtures';
import { type ImportDatasetPage } from './import-dataset-page';
import {
    DATASET_FILENAME,
    deleteStagedDatasetHandler,
    getMockedImportJob,
    getMockedPrepareJob,
    IMPORT_JOB_ID,
    jobStatusStreamHandler,
    PREPARE_JOB_ID,
    STAGED_DATASET_ID,
    stagedDatasetWithMetadata,
} from './utils';

const openCreateFromDatasetDialog = async (page: Page) => {
    await page.getByRole('button', { name: 'Create from dataset' }).click();
};

const uploadAndWaitForPreparation = async (importDatasetPage: ImportDatasetPage) => {
    await importDatasetPage.loadZipFile(DATASET_FILENAME);
    await expect(importDatasetPage.getPreparingStatus()).toBeVisible();
};

const selectTaskTypeAndProceed = async (importDatasetPage: ImportDatasetPage) => {
    const dialog = importDatasetPage.getDialog();
    await expect(dialog.getByRole('button', { name: 'Object detection (Recommended) Task' })).toBeVisible();
    await dialog.getByRole('button', { name: 'Next' }).click();
};

const submitImport = async (importDatasetPage: ImportDatasetPage) => {
    const dialog = importDatasetPage.getDialog();
    await dialog.getByRole('button', { name: 'Create' }).click();
    await expect(dialog).toBeHidden();
};

test.describe('Import dataset as new project', () => {
    test.beforeEach(({ network }) => {
        network.use(
            http.get('/api/staged_datasets/{staged_dataset_id}', () => {
                return HttpResponse.json(stagedDatasetWithMetadata, { status: 200 });
            }),
            http.post('/api/staged_datasets:from-upload', () => {
                return HttpResponse.json(getMockedStagedDataset({ id: STAGED_DATASET_ID }), { status: 201 });
            }),
            http.post('/api/jobs', async ({ request }) => {
                const body = await request.json();
                if (body.job_type === 'import_dataset_as_new_project') {
                    return HttpResponse.json(getMockedImportJob('import_dataset_as_new_project'), { status: 201 });
                }
                return HttpResponse.json(getMockedPrepareJob(), { status: 201 });
            })
        );
    });

    test('import dataset as new project with default settings', async ({ page, network, importDatasetPage }) => {
        const importingJob = getMockedImportJob('import_dataset_as_new_project', {
            progress: 60,
            message: 'Importing progress...',
        });

        const preparingJob = getMockedPrepareJob();
        const preparedJob = getMockedPrepareJob({ status: 'DONE', progress: 100, message: 'Preparation completed' });
        const importedJob = getMockedImportJob('import_dataset_as_new_project', {
            status: 'DONE',
            progress: 100,
            message: 'Import completed',
        });
        const { handler: deleteHandler, getDeletedId } = deleteStagedDatasetHandler();

        network.use(
            http.get('/api/jobs/{job_id}', ({ params }) => {
                const job = params.job_id === IMPORT_JOB_ID ? importingJob : preparingJob;
                return HttpResponse.json(job, { status: 200 });
            }),
            jobStatusStreamHandler({
                [PREPARE_JOB_ID]: [preparingJob, preparedJob],
                [IMPORT_JOB_ID]: [importingJob, importedJob],
            }),
            deleteHandler
        );

        await page.goto('/projects');

        await test.step('Open create project from dataset dialog', async () => {
            await openCreateFromDatasetDialog(page);
        });

        await test.step('Upload dataset zip file', async () => {
            await importDatasetPage.loadZipFile(DATASET_FILENAME);
        });

        await test.step('Verify preparation progress', async () => {
            await expect(importDatasetPage.getPreparingStatus()).toBeVisible();
        });

        await test.step('Select task type and proceed', async () => {
            const dialog = importDatasetPage.getDialog();

            await expect(dialog.getByLabel('Project name')).toBeVisible();
            await selectTaskTypeAndProceed(importDatasetPage);
        });

        await test.step('Verify label mapping and create project', async () => {
            const dialog = importDatasetPage.getDialog();

            await expect(importDatasetPage.getStatisticsHeading()).toBeVisible();
            await expect(importDatasetPage.getLabelMappingHeading()).toBeVisible();

            for (const label of stagedDatasetWithMetadata.metadata?.labels ?? []) {
                await expect(dialog.getByText(label)).toBeVisible();
            }

            await expect(importDatasetPage.getIncludeUnannotatedCheckbox()).toBeChecked();
            await submitImport(importDatasetPage);
        });

        await test.step('Verify import job progress', async () => {
            await expect(importDatasetPage.getImportStatusText(DATASET_FILENAME, 'processing')).toBeVisible();
            await expect(page.getByText(String(importingJob.message))).toBeVisible();
        });

        await test.step('Show success toast and removes the staged dataset', async () => {
            await expect(page.getByText(`Dataset ${DATASET_FILENAME} 16 B`)).toBeVisible();
            await expect.poll(() => getDeletedId()).toBe(STAGED_DATASET_ID);
        });
    });

    test('cancel prepare job removes staged files', async ({ page, network, importDatasetPage }) => {
        const runningPrepareJob = getMockedPrepareJob();
        const { handler: deleteHandler, getDeletedId } = deleteStagedDatasetHandler();

        network.use(
            http.post('/api/jobs', async () => {
                return HttpResponse.json(getMockedPrepareJob(), { status: 201 });
            }),
            http.get('/api/jobs/{job_id}', () => {
                return HttpResponse.json(runningPrepareJob, { status: 200 });
            }),
            http.post('/api/jobs/{job_id}:cancel', () => {
                return HttpResponse.json(getMockedJob({ ...runningPrepareJob, status: 'CANCELLED' }), { status: 200 });
            }),
            deleteHandler
        );

        await page.goto('/projects');

        await test.step('Open create project from dataset dialog', async () => {
            await openCreateFromDatasetDialog(page);
        });

        await test.step('Upload dataset zip file', async () => {
            await importDatasetPage.loadZipFile(DATASET_FILENAME);
        });

        await test.step('Cancel the prepare job from dialog', async () => {
            await expect(importDatasetPage.getPreparingStatus()).toBeVisible();
            await importDatasetPage.cancelPrepareJobInDialog();

            await expect(importDatasetPage.getDialog()).toBeHidden();
            await expect.poll(() => getDeletedId()).toBe(STAGED_DATASET_ID);
        });
    });

    test('cancel import job removes staged files', async ({ page, network, importDatasetPage }) => {
        const importingJob = getMockedImportJob('import_dataset_as_new_project', {
            progress: 60,
            message: 'Importing progress...',
        });

        const preparingJob = getMockedPrepareJob();
        const preparedJob = getMockedPrepareJob({ status: 'DONE', progress: 100, message: 'Preparation completed' });
        const { handler: deleteHandler, getDeletedId } = deleteStagedDatasetHandler();

        network.use(
            http.get('/api/jobs/{job_id}', ({ params }) => {
                const job = params.job_id === IMPORT_JOB_ID ? importingJob : preparingJob;
                return HttpResponse.json(job, { status: 200 });
            }),
            jobStatusStreamHandler({
                [PREPARE_JOB_ID]: [preparingJob, preparedJob],
                [IMPORT_JOB_ID]: [importingJob],
            }),
            http.post('/api/jobs/{job_id}:cancel', () => {
                return HttpResponse.json(getMockedJob({ ...importingJob, status: 'CANCELLED' }), { status: 200 });
            }),
            deleteHandler
        );

        await page.goto('/projects');

        await test.step('Open create project from dataset dialog', async () => {
            await openCreateFromDatasetDialog(page);
        });

        await test.step('Upload and complete preparation', async () => {
            await uploadAndWaitForPreparation(importDatasetPage);
        });

        await test.step('Select task type and proceed', async () => {
            await selectTaskTypeAndProceed(importDatasetPage);
        });

        await test.step('Submit import', async () => {
            await submitImport(importDatasetPage);
        });

        await test.step('Cancel the running import job', async () => {
            await expect(importDatasetPage.getImportStatusText(DATASET_FILENAME, 'processing')).toBeVisible();
            await importDatasetPage.cancelJobFromStatusCard();

            await expect(importDatasetPage.getImportStatusText(DATASET_FILENAME, 'processing')).toBeHidden();
            await expect.poll(() => getDeletedId()).toBe(STAGED_DATASET_ID);
        });
    });

    test('failed prepare job closes dialog', async ({ page, network, importDatasetPage }) => {
        const errorData = { message: 'An error occurred during preparation.', error: 'error test' };
        const failedPrepareJob = getMockedPrepareJob({ status: 'FAILED', ...errorData });
        const { handler: deleteHandler, getDeletedId } = deleteStagedDatasetHandler();

        network.use(
            http.post('/api/jobs', async () => {
                return HttpResponse.json(getMockedPrepareJob(), { status: 201 });
            }),
            http.get('/api/jobs/{job_id}', () => {
                return HttpResponse.json(failedPrepareJob, { status: 200 });
            }),
            deleteHandler
        );

        await page.goto('/projects');

        await test.step('Open create project from dataset dialog', async () => {
            await openCreateFromDatasetDialog(page);
        });

        await test.step('Upload dataset zip file', async () => {
            await importDatasetPage.loadZipFile(DATASET_FILENAME);
        });

        await test.step('Prepare job fails and dialog closes', async () => {
            await expect(importDatasetPage.getDialog()).toBeHidden();
            await page.getByLabel('Technical details of the job failure').click();
            await expect(page.getByText(errorData.error, { exact: true })).toBeVisible();
            await expect(page.getByText(errorData.message, { exact: true })).toBeVisible();
        });

        await test.step('Close error notification removes staged file', async () => {
            await importDatasetPage.closeImportStatus();

            await expect(page.getByText('Import failed')).toBeHidden();
            await expect.poll(() => getDeletedId()).toBe(STAGED_DATASET_ID);
        });
    });

    test('failed import job shows error and closing removes staged file', async ({
        page,
        network,
        importDatasetPage,
    }) => {
        const errorData = { message: 'An error occurred during preparation.', error: 'error test' };
        const failedImportJob = getMockedImportJob('import_dataset_as_new_project', { status: 'FAILED', ...errorData });

        const preparingJob = getMockedPrepareJob();
        const preparedJob = getMockedPrepareJob({ status: 'DONE', progress: 100, message: 'Preparation completed' });
        const { handler: deleteHandler, getDeletedId } = deleteStagedDatasetHandler();

        network.use(
            http.get('/api/jobs/{job_id}', ({ params }) => {
                const job = params.job_id === IMPORT_JOB_ID ? failedImportJob : preparingJob;
                return HttpResponse.json(job, { status: 200 });
            }),
            jobStatusStreamHandler({
                [PREPARE_JOB_ID]: [preparingJob, preparedJob],
                [IMPORT_JOB_ID]: [failedImportJob],
            }),
            deleteHandler
        );

        await page.goto('/projects');

        await test.step('Open create project from dataset dialog', async () => {
            await openCreateFromDatasetDialog(page);
        });

        await test.step('Upload and complete preparation', async () => {
            await uploadAndWaitForPreparation(importDatasetPage);
        });

        await test.step('Select task type and proceed', async () => {
            await selectTaskTypeAndProceed(importDatasetPage);
        });

        await test.step('Submit import', async () => {
            await submitImport(importDatasetPage);
        });

        await test.step('Verify error notification is shown', async () => {
            await page.getByLabel('Technical details of the job failure').click();
            await expect(page.getByText(errorData.error, { exact: true })).toBeVisible();
            await expect(page.getByText(errorData.message, { exact: true })).toBeVisible();
        });

        await test.step('Close error notification removes staged file', async () => {
            await importDatasetPage.closeImportStatus();

            await expect(page.getByText('Import failed')).toBeHidden();
            await expect.poll(() => getDeletedId()).toBe(STAGED_DATASET_ID);
        });
    });
});
