// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedJob } from 'mocks/mock-job';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';

import { expect, http, test } from '../fixtures';

const PROJECT_A = getMockedProject({ id: 'project-a', name: 'Alpha' });
const PROJECT_B = getMockedProject({ id: 'project-b', name: 'Beta' });

test.describe('Jobs dialog', () => {
    test('opens in "All projects" scope from the project list', async ({ jobsDialogPage, network }) => {
        network.use(
            http.get('/api/projects', () => HttpResponse.json([PROJECT_A, PROJECT_B])),
            http.get('/api/jobs', () => HttpResponse.json([]))
        );

        await jobsDialogPage.gotoProjectList();
        await jobsDialogPage.openDialog();

        await expect(jobsDialogPage.getProjectPicker()).toHaveText('All projects');
    });

    test('opens scoped to the current project from inside a project', async ({ jobsDialogPage, network }) => {
        network.use(
            http.get('/api/projects', () => HttpResponse.json([PROJECT_A, PROJECT_B])),
            http.get('/api/jobs', () => HttpResponse.json([]))
        );

        await jobsDialogPage.gotoProject('project-a');
        await jobsDialogPage.openDialog();

        await expect(jobsDialogPage.getProjectPicker()).toHaveText(`${PROJECT_A.name} (current)`);
    });

    test('switching the project picker re-filters the list', async ({ jobsDialogPage, network }) => {
        const jobInA = getMockedJob({
            job_id: 'job-a',
            metadata: {
                project: { id: 'project-a' },
                model: {
                    id: 'model-a',
                    name: 'alpha model',
                    architecture: 'Custom_Object_Detection_Gen3_ATSS',
                    parent_revision_id: null,
                    dataset_revision_id: 'dataset-1',
                },
                device: { type: 'cpu', name: 'CPU' },
            },
        });
        const jobInB = getMockedJob({
            job_id: 'job-b',
            metadata: {
                project: { id: 'project-b' },
                model: {
                    id: 'model-b',
                    name: 'beta model',
                    architecture: 'Custom_Object_Detection_Gen3_ATSS',
                    parent_revision_id: null,
                    dataset_revision_id: 'dataset-1',
                },
                device: { type: 'cpu', name: 'CPU' },
            },
        });

        network.use(
            http.get('/api/projects', () => HttpResponse.json([PROJECT_A, PROJECT_B])),
            http.get('/api/jobs', () => HttpResponse.json([jobInA, jobInB]))
        );

        await jobsDialogPage.gotoProjectList();
        await jobsDialogPage.openDialog();

        await expect(jobsDialogPage.getJobByName('alpha model')).toBeVisible();
        await expect(jobsDialogPage.getJobByName('beta model')).toBeVisible();

        await jobsDialogPage.selectProject('Beta');

        await expect(jobsDialogPage.getProjectPicker()).toHaveText('Beta');
        await expect(jobsDialogPage.getJobByName('beta model')).toBeVisible();
        await expect(jobsDialogPage.getJobByName('alpha model')).toBeHidden();
    });

    test('switching status tabs filters the list and hides zero-count tabs', async ({ jobsDialogPage, network }) => {
        const runningJob = getMockedJob({
            job_id: 'running',
            status: 'RUNNING',
            started_at: '2026-01-01T00:00:00Z',
            metadata: {
                project: { id: '7b073838-99d3-42ff-9018-4e901eb047fc' },
                model: {
                    id: 'running-model',
                    name: 'running model',
                    architecture: 'Custom_Object_Detection_Gen3_ATSS',
                    parent_revision_id: null,
                    dataset_revision_id: 'dataset-1',
                },
                device: { type: 'cpu', name: 'CPU' },
            },
        });
        const pendingJob = getMockedJob({
            job_id: 'pending',
            status: 'PENDING',
            started_at: null,
            metadata: {
                project: { id: '7b073838-99d3-42ff-9018-4e901eb047fc' },
                model: {
                    id: 'pending-model',
                    name: 'pending model',
                    architecture: 'Custom_Object_Detection_Gen3_ATSS',
                    parent_revision_id: null,
                    dataset_revision_id: 'dataset-1',
                },
                device: { type: 'cpu', name: 'CPU' },
            },
        });

        network.use(
            http.get('/api/projects', () => HttpResponse.json([PROJECT_A])),
            http.get('/api/jobs', () => HttpResponse.json([runningJob, pendingJob]))
        );

        await jobsDialogPage.gotoProjectList();
        await jobsDialogPage.openDialog();

        await expect(jobsDialogPage.getJobByName('running model')).toBeVisible();
        await expect(jobsDialogPage.getJobByName('pending model')).toBeVisible();
        await expect(jobsDialogPage.getTab('All')).toHaveText('All2');
        await expect(jobsDialogPage.getTab('Failed')).toHaveText('Failed');

        await jobsDialogPage.selectTab('Running');

        await expect(jobsDialogPage.getJobByName('running model')).toBeVisible();
        await expect(jobsDialogPage.getJobByName('pending model')).toBeHidden();
    });

    test('shows the status badge next to the job name only on the All tab', async ({ jobsDialogPage, network }) => {
        const finishedJob = getMockedJob({
            job_id: 'done',
            status: 'DONE',
            started_at: '2026-01-01T00:00:00Z',
            finished_at: '2026-01-01T01:00:00Z',
            metadata: {
                project: { id: '7b073838-99d3-42ff-9018-4e901eb047fc' },
                model: {
                    id: 'done-model',
                    name: 'finished model',
                    architecture: 'Custom_Object_Detection_Gen3_ATSS',
                    parent_revision_id: null,
                    dataset_revision_id: 'dataset-1',
                },
                device: { type: 'cpu', name: 'CPU' },
            },
        });

        network.use(
            http.get('/api/projects', () => HttpResponse.json([PROJECT_A])),
            http.get('/api/jobs', () => HttpResponse.json([finishedJob]))
        );

        await jobsDialogPage.gotoProjectList();
        await jobsDialogPage.openDialog();

        await expect(jobsDialogPage.getJobByName('finished model')).toBeVisible();
        await expect(jobsDialogPage.getStatusBadge('Completed')).toBeVisible();

        await jobsDialogPage.selectTab('Finished');

        await expect(jobsDialogPage.getJobByName('finished model')).toBeVisible();
        await expect(jobsDialogPage.getStatusBadge('Completed')).toBeHidden();
    });

    test('cancels a running job through the actions menu, not offered for a finished job', async ({
        jobsDialogPage,
        network,
        page,
    }) => {
        let hasCancelled = false;

        const runningJob = getMockedJob({
            job_id: 'running',
            status: 'RUNNING',
            started_at: '2026-01-02T00:00:00Z',
            metadata: {
                project: { id: '7b073838-99d3-42ff-9018-4e901eb047fc' },
                model: {
                    id: 'running-model',
                    name: 'running model',
                    architecture: 'Custom_Object_Detection_Gen3_ATSS',
                    parent_revision_id: null,
                    dataset_revision_id: 'dataset-1',
                },
                device: { type: 'cpu', name: 'CPU' },
            },
        });
        const finishedJob = getMockedJob({
            job_id: 'done',
            status: 'DONE',
            started_at: '2026-01-01T00:00:00Z',
            finished_at: '2026-01-01T01:00:00Z',
            metadata: {
                project: { id: '7b073838-99d3-42ff-9018-4e901eb047fc' },
                model: {
                    id: 'done-model',
                    name: 'done model',
                    architecture: 'Custom_Object_Detection_Gen3_ATSS',
                    parent_revision_id: null,
                    dataset_revision_id: 'dataset-1',
                },
                device: { type: 'cpu', name: 'CPU' },
            },
        });

        network.use(
            http.get('/api/projects', () => HttpResponse.json([PROJECT_A])),
            http.get('/api/jobs', () => HttpResponse.json(hasCancelled ? [finishedJob] : [runningJob, finishedJob])),
            http.post('/api/jobs/{job_id}:cancel', () => {
                hasCancelled = true;

                return HttpResponse.json(null, { status: 204 });
            })
        );

        await jobsDialogPage.gotoProjectList();
        await jobsDialogPage.openDialog();

        await jobsDialogPage.openActionsMenu('done model');
        await expect(jobsDialogPage.getViewLogsMenuItem()).toBeVisible();
        await expect(jobsDialogPage.getCancelMenuItem()).toBeHidden();
        await page.keyboard.press('Escape');

        await jobsDialogPage.cancelJob('running model');

        await expect(jobsDialogPage.getJobByName('running model')).toBeHidden();
        await expect(jobsDialogPage.getJobByName('done model')).toBeVisible();
        await expect(jobsDialogPage.getTab('Running')).toHaveText('Running');
    });

    test('opening logs returns to the jobs dialog with the same tab and scope', async ({ jobsDialogPage, network }) => {
        const runningJob = getMockedJob({
            job_id: 'running',
            status: 'RUNNING',
            started_at: '2026-01-01T00:00:00Z',
            metadata: {
                project: { id: '7b073838-99d3-42ff-9018-4e901eb047fc' },
                model: {
                    id: 'running-model',
                    name: 'running model',
                    architecture: 'Custom_Object_Detection_Gen3_ATSS',
                    parent_revision_id: null,
                    dataset_revision_id: 'dataset-1',
                },
                device: { type: 'cpu', name: 'CPU' },
            },
        });

        network.use(
            http.get('/api/projects', () => HttpResponse.json([PROJECT_A, PROJECT_B])),
            http.get('/api/jobs', () => HttpResponse.json([runningJob])),
            http.get('/api/jobs/{job_id}/logs', () => {
                return new HttpResponse(':ok\n\n', {
                    status: 200,
                    headers: { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' },
                });
            })
        );

        await jobsDialogPage.gotoProjectList();
        await jobsDialogPage.openDialog();
        await jobsDialogPage.selectTab('Running');

        await jobsDialogPage.viewLogs('running model');

        await expect(jobsDialogPage.getLogsHeading()).toBeVisible();

        await jobsDialogPage.closeLogsDialog();

        await expect(jobsDialogPage.getLogsHeading()).toBeHidden();
        await expect(jobsDialogPage.getHeading('Jobs')).toBeVisible();
        await expect(jobsDialogPage.getProjectPicker()).toHaveText('All projects');
        await expect(jobsDialogPage.getTab('Running')).toHaveAttribute('aria-selected', 'true');
    });

    test('shows the bell corner dot when a job is running', async ({ jobsDialogPage, network }) => {
        network.use(
            http.get('/api/projects', () => HttpResponse.json([PROJECT_A])),
            http.get('/api/jobs', () => HttpResponse.json([getMockedJob({ job_id: 'running', status: 'RUNNING' })]))
        );

        await jobsDialogPage.gotoProjectList();

        await expect(jobsDialogPage.getJobsButtonDot()).toBeVisible();
    });

    test('hides the bell corner dot when jobs are only scheduled or finished', async ({ jobsDialogPage, network }) => {
        network.use(
            http.get('/api/projects', () => HttpResponse.json([PROJECT_A])),
            http.get('/api/jobs', () =>
                HttpResponse.json([
                    getMockedJob({ job_id: 'pending', status: 'PENDING', started_at: null }),
                    getMockedJob({
                        job_id: 'done',
                        status: 'DONE',
                        started_at: '2026-01-01T00:00:00Z',
                        finished_at: '2026-01-01T01:00:00Z',
                    }),
                ])
            )
        );

        await jobsDialogPage.gotoProjectList();

        await expect(jobsDialogPage.getJobsButtonDot()).toBeHidden();
    });
});
