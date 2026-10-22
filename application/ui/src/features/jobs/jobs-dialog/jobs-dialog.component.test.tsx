// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { TrainJob } from '@/api/types';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedQuantizeJob, getMockedTrainJob } from 'mocks/mock-job';
import { getMockedModelArchitecture } from 'mocks/mock-model';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../api/utils';
import { paths } from '../../../constants/paths';
import { server } from '../../../msw-node-setup';
import { downloadFile } from '../../../platform/download-file';
import { ALL_PROJECTS, type ModelJob } from '../utils';
import { JobsDialog } from './jobs-dialog.component';

rs.mock('../../../platform/download-file', () => ({
    downloadFile: rs.fn(),
}));

const ARCH_ID = 'Object_Detection_Deim_DFine_L';
const PROJECT_A = getMockedProject({ id: 'project-a', name: 'Alpha' });
const PROJECT_B = getMockedProject({ id: 'project-b', name: 'Beta' });

const trainJob = (overrides: Partial<TrainJob> & { job_id: string }): TrainJob =>
    getMockedTrainJob({
        metadata: {
            project: { id: 'project-a' },
            model: {
                id: `${overrides.job_id}-model`,
                name: `${overrides.job_id} model`,
                architecture: ARCH_ID,
                parent_revision_id: null,
                dataset_revision_id: 'rev-1',
            },
            device: { type: 'cpu', name: 'CPU' },
        },
        ...overrides,
    });

const mockProjectsAndArchitectures = (projects = [PROJECT_A, PROJECT_B]) => {
    server.use(
        http.get('/api/projects', () => HttpResponse.json(projects)),
        http.get('/api/model_architectures', () =>
            HttpResponse.json({
                model_architectures: [getMockedModelArchitecture({ id: ARCH_ID, name: 'Deim-DFine-L' })],
                top_picks: { balance: ARCH_ID, speed: ARCH_ID, accuracy: ARCH_ID },
            })
        )
    );
};

const mockJobs = (jobs: ModelJob[]) => {
    server.use(http.get('/api/jobs', () => HttpResponse.json(jobs)));
};

const renderDialog = (props: Partial<Parameters<typeof JobsDialog>[0]> = {}) =>
    render(<JobsDialog initialProjectId={ALL_PROJECTS} onClose={rs.fn()} {...props} />);

const openJobActions = async (modelName: string) =>
    userEvent.click(await screen.findByRole('button', { name: `Job actions for ${modelName}` }));

const getJobProgressBar = () => screen.queryByRole('progressbar', { name: 'Job progress' });
const getLoadingSpinner = () => screen.queryByRole('progressbar', { name: 'Loading...' });

describe('JobsDialog', () => {
    beforeEach(() => {
        mockProjectsAndArchitectures();
    });

    it('shows tab counts scoped to the selected project, and hides tabs with a zero count', async () => {
        mockJobs([
            trainJob({ job_id: 'running', status: 'RUNNING', progress: 50, started_at: '2026-01-01T00:00:00Z' }),
            trainJob({ job_id: 'pending', status: 'PENDING', started_at: null }),
            trainJob({
                job_id: 'done-other-project',
                status: 'DONE',
                started_at: '2026-01-01T00:00:00Z',
                finished_at: '2026-01-01T01:00:00Z',
                metadata: {
                    project: { id: 'project-b' },
                    model: {
                        id: 'done-model',
                        name: 'done model',
                        architecture: ARCH_ID,
                        parent_revision_id: null,
                        dataset_revision_id: 'rev-1',
                    },
                    device: { type: 'cpu', name: 'CPU' },
                },
            }),
        ]);

        renderDialog();

        await screen.findByText('running model');

        expect(screen.getByRole('tab', { name: /^All\b/ })).toHaveTextContent('All3');
        expect(screen.getByRole('tab', { name: /^Running\b/ })).toHaveTextContent('Running1');
        expect(screen.getByRole('tab', { name: /^Queued\b/ })).toHaveTextContent('Queued1');
        expect(screen.getByRole('tab', { name: /^Finished\b/ })).toHaveTextContent('Finished1');
        expect(screen.getByRole('tab', { name: 'Cancelled' })).toHaveTextContent('Cancelled');
        expect(screen.getByRole('tab', { name: 'Failed' })).toHaveTextContent('Failed');
    });

    it('filters the list by project scope', async () => {
        mockJobs([
            trainJob({ job_id: 'in-a', status: 'RUNNING', started_at: '2026-01-01T00:00:00Z' }),
            trainJob({
                job_id: 'in-b',
                status: 'RUNNING',
                started_at: '2026-01-01T00:00:00Z',
                metadata: {
                    project: { id: 'project-b' },
                    model: {
                        id: 'in-b-model',
                        name: 'in-b model',
                        architecture: ARCH_ID,
                        parent_revision_id: null,
                        dataset_revision_id: 'rev-1',
                    },
                    device: { type: 'cpu', name: 'CPU' },
                },
            }),
        ]);

        renderDialog({ initialProjectId: 'project-a', currentProjectId: 'project-a' });

        expect(await screen.findByRole('button', { name: /Filter jobs by project/ })).toHaveTextContent(PROJECT_A.name);
        expect(await screen.findByText('in-a model')).toBeInTheDocument();
        expect(screen.queryByText('in-b model')).not.toBeInTheDocument();
    });

    it('shows queued-job waiting state, spinner, and cancel action only', async () => {
        mockJobs([trainJob({ job_id: 'pending', status: 'PENDING', started_at: null, message: null })]);

        renderDialog({ initialProjectId: ALL_PROJECTS });

        expect(await screen.findByRole('link', { name: 'Alpha' })).toBeInTheDocument();
        expect(screen.getByText('Waiting to start...')).toBeInTheDocument();
        expect(getLoadingSpinner()).toBeInTheDocument();
        expect(getJobProgressBar()).not.toBeInTheDocument();
        await openJobActions('pending model');
        expect(screen.getByRole('menuitem', { name: 'Cancel' })).toBeInTheDocument();
        expect(screen.queryByRole('menuitem', { name: 'View logs' })).not.toBeInTheDocument();
    });

    it('links every job project name to its models page', async () => {
        const jobs = [
            trainJob({ job_id: 'pending', status: 'PENDING', started_at: null }),
            trainJob({ job_id: 'running', status: 'RUNNING', started_at: '2026-01-01T00:00:00Z' }),
            trainJob({ job_id: 'cancelling', status: 'CANCELLING', started_at: '2026-01-01T00:00:00Z' }),
            trainJob({ job_id: 'done', status: 'DONE', started_at: '2026-01-01T00:00:00Z' }),
            trainJob({ job_id: 'failed', status: 'FAILED', started_at: '2026-01-01T00:00:00Z' }),
            trainJob({ job_id: 'cancelled', status: 'CANCELLED', started_at: '2026-01-01T00:00:00Z' }),
        ];
        mockJobs(jobs);
        renderDialog({ initialProjectId: ALL_PROJECTS });

        const projectLinks = await screen.findAllByRole('link', { name: 'Alpha' });

        expect(projectLinks).toHaveLength(jobs.length);
        projectLinks.forEach((link) =>
            expect(link).toHaveAttribute('href', paths.project.models({ projectId: 'project-a' }))
        );
    });

    it('does not render a job progress bar for queued job', async () => {
        mockJobs([trainJob({ job_id: 'pending', status: 'PENDING', started_at: null })]);

        renderDialog();

        await screen.findByText('pending model');
        expect(getJobProgressBar()).not.toBeInTheDocument();
    });

    it('shows running-job spinner and 0% progress bar', async () => {
        mockJobs([
            trainJob({
                job_id: 'running-zero',
                status: 'RUNNING',
                progress: 0,
                started_at: '2026-01-01T00:00:00Z',
            }),
        ]);

        renderDialog();

        await screen.findByText('running-zero model');
        expect(getLoadingSpinner()).toBeInTheDocument();
        expect(screen.queryByText(/\d+%/)).not.toBeInTheDocument();
        expect(getJobProgressBar()).toHaveAttribute('aria-valuenow', '0');
        await openJobActions('running-zero model');
        expect(screen.getByRole('menuitem', { name: 'View logs' })).toBeInTheDocument();
        expect(screen.getByRole('menuitem', { name: 'Cancel' })).toBeInTheDocument();
    });

    it('shows running-job percentage and progress bar above 0%', async () => {
        mockJobs([
            trainJob({ job_id: 'running', status: 'RUNNING', progress: 42, started_at: '2026-01-01T00:00:00Z' }),
        ]);

        renderDialog();

        await screen.findByText('running model');
        expect(screen.getByText(/42%/)).toBeInTheDocument();
        expect(getLoadingSpinner()).not.toBeInTheDocument();
        expect(getJobProgressBar()).toHaveAttribute('aria-valuenow', '42');
        await openJobActions('running model');
        expect(screen.getByRole('menuitem', { name: 'View logs' })).toBeInTheDocument();
        expect(screen.getByRole('menuitem', { name: 'Cancel' })).toBeInTheDocument();
    });

    it('shows cancelling-job running metadata and View logs only', async () => {
        mockJobs([
            trainJob({
                job_id: 'job',
                status: 'CANCELLING',
                progress: 25,
                started_at: '2026-01-01T00:00:00Z',
                finished_at: null,
            }),
        ]);

        renderDialog();

        await screen.findByText('job model');
        expect(screen.getByText(/Started.*Elapsed/)).toBeInTheDocument();
        expect(getJobProgressBar()).toHaveAttribute('aria-valuenow', '25');
        await openJobActions('job model');
        expect(screen.getByRole('menuitem', { name: 'View logs' })).toBeInTheDocument();
        expect(screen.queryByRole('menuitem', { name: 'Cancel' })).not.toBeInTheDocument();
    });

    it('confirms and cancels a running job through the actions menu', async () => {
        const cancelSpy = rs.fn();
        mockJobs([trainJob({ job_id: 'running', status: 'RUNNING', started_at: '2026-01-01T00:00:00Z' })]);
        server.use(
            http.post('/api/jobs/{job_id}:cancel', () => {
                cancelSpy();
                return HttpResponse.json(null, { status: 204 });
            })
        );

        renderDialog();

        await openJobActions('running model');
        await userEvent.click(screen.getByRole('menuitem', { name: 'Cancel' }));

        expect(await screen.findByRole('alertdialog', { name: 'Cancel job' })).toBeInTheDocument();
        await userEvent.click(screen.getByRole('button', { name: 'Cancel job' }));
        await rs.waitFor(() => expect(cancelSpy).toHaveBeenCalled());
    });

    it('opens running-job logs and returns to the jobs dialog on close, tab unchanged', async () => {
        mockJobs([trainJob({ job_id: 'running', status: 'RUNNING', started_at: '2026-01-01T00:00:00Z' })]);
        renderDialog();

        await screen.findByText('running model');
        await userEvent.click(screen.getByRole('tab', { name: /^Running\b/ }));
        await openJobActions('running model');
        await userEvent.click(screen.getByRole('menuitem', { name: 'View logs' }));

        expect(await screen.findByText('Training Logs')).toBeInTheDocument();
        await userEvent.click(screen.getByRole('button', { name: 'Close dialog' }));
        await waitFor(() => expect(screen.queryByText('Training Logs')).not.toBeInTheDocument());
        expect(screen.getByRole('button', { name: /Filter jobs by project/ })).toHaveTextContent('All projects');
        expect(screen.getByRole('tab', { name: /^Running\b/, selected: true })).toBeInTheDocument();
    });

    it('downloads the job logs as plain text from the logs dialog', async () => {
        mockJobs([trainJob({ job_id: 'running', status: 'RUNNING', started_at: '2026-01-01T00:00:00Z' })]);
        renderDialog();

        await screen.findByText('running model');
        await openJobActions('running model');
        await userEvent.click(screen.getByRole('menuitem', { name: 'View logs' }));
        await userEvent.click(await screen.findByRole('button', { name: 'Download logs' }));

        expect(downloadFile).toHaveBeenCalledWith(
            expect.stringMatching(/\/api\/jobs\/running\/logs\?format=text$/),
            'job-logs-running.log',
            'Training logs download started'
        );
    });

    it('shows finished-job metadata and view logs only', async () => {
        mockJobs([
            trainJob({
                job_id: 'done',
                status: 'DONE',
                progress: 100,
                started_at: '2026-01-01T00:00:00Z',
                finished_at: '2026-01-01T01:00:00Z',
                message: 'Completed cleanly',
            }),
        ]);

        renderDialog();

        await screen.findByText('done model');
        expect(screen.getByText(/Duration 1h 0m 0s.*Finished/)).toBeInTheDocument();
        expect(screen.getByText('Completed cleanly')).toBeInTheDocument();
        expect(getLoadingSpinner()).not.toBeInTheDocument();
        expect(getJobProgressBar()).not.toBeInTheDocument();
        await openJobActions('done model');
        expect(screen.getByRole('menuitem', { name: 'View logs' })).toBeInTheDocument();
        expect(screen.queryByRole('menuitem', { name: 'Cancel' })).not.toBeInTheDocument();
    });

    it('shows the row status badge only on the All tab for finished jobs', async () => {
        mockJobs([
            trainJob({
                job_id: 'done',
                status: 'DONE',
                started_at: '2026-01-01T00:00:00Z',
                finished_at: '2026-01-01T01:00:00Z',
            }),
        ]);

        renderDialog();

        await screen.findByText('done model');
        expect(screen.getByText('Completed')).toBeInTheDocument();
        await userEvent.click(screen.getByRole('tab', { name: /^Finished\b/ }));
        expect(screen.getByText('done model')).toBeInTheDocument();
        expect(screen.queryByText('Completed')).not.toBeInTheDocument();
    });

    it('shows failed-job metadata and view logs only', async () => {
        mockJobs([
            trainJob({
                job_id: 'failed',
                status: 'FAILED',
                started_at: '2026-01-01T00:00:00Z',
                finished_at: '2026-01-01T00:05:00Z',
                error: 'CUDA out of memory',
                message: 'Training stopped',
            }),
        ]);

        renderDialog();

        expect(await screen.findByText('failed model')).toBeInTheDocument();
        expect(screen.getByText(/^Failed .+/)).toBeInTheDocument();
        expect(screen.getByText('Training stopped')).toBeInTheDocument();
        expect(getLoadingSpinner()).not.toBeInTheDocument();
        expect(getJobProgressBar()).not.toBeInTheDocument();
        expect(screen.queryByText('CUDA out of memory')).not.toBeInTheDocument();
        await openJobActions('failed model');
        expect(screen.getByRole('menuitem', { name: 'View logs' })).toBeInTheDocument();
        expect(screen.queryByRole('menuitem', { name: 'Cancel' })).not.toBeInTheDocument();
    });

    it('shows no waiting state, spinner, progress bar, or actions for a cancelled job that never started', async () => {
        mockJobs([
            trainJob({
                job_id: 'cancelled-before-start',
                status: 'CANCELLED',
                progress: 0,
                started_at: null,
            }),
        ]);

        renderDialog();

        await screen.findByText('cancelled-before-start model');
        expect(screen.queryByText(/Waiting to start/)).not.toBeInTheDocument();
        expect(getLoadingSpinner()).not.toBeInTheDocument();
        expect(
            screen.queryByRole('button', { name: 'Job actions for cancelled-before-start model' })
        ).not.toBeInTheDocument();
        expect(getJobProgressBar()).not.toBeInTheDocument();
    });

    it('shows cancelled-job date and view logs only when the job started', async () => {
        mockJobs([
            trainJob({
                job_id: 'cancelled',
                status: 'CANCELLED',
                progress: 0,
                started_at: '2026-01-01T00:00:00Z',
                finished_at: '2026-01-01T00:05:00Z',
            }),
        ]);

        renderDialog();

        await screen.findByText('cancelled model');
        expect(screen.getByText(/^Cancelled .+/)).toBeInTheDocument();
        expect(getLoadingSpinner()).not.toBeInTheDocument();
        expect(getJobProgressBar()).not.toBeInTheDocument();
        await openJobActions('cancelled model');
        expect(screen.getByRole('menuitem', { name: 'View logs' })).toBeInTheDocument();
        expect(screen.queryByRole('menuitem', { name: 'Cancel' })).not.toBeInTheDocument();
    });

    it('shows the empty state illustrated message and hides the table header when a tab has no jobs', async () => {
        mockJobs([trainJob({ job_id: 'running', status: 'RUNNING', started_at: '2026-01-01T00:00:00Z' })]);
        renderDialog();

        await userEvent.click(await screen.findByRole('tab', { name: 'Failed' }));

        expect(await screen.findByRole('heading', { name: 'No failed jobs' })).toBeInTheDocument();
        expect(screen.queryByText('Architecture')).not.toBeInTheDocument();
    });

    it('shows "—" for a quantize job device', async () => {
        server.use(
            http.get('/api/jobs', () =>
                HttpResponse.json([
                    getMockedQuantizeJob({
                        job_id: 'quantize',
                        status: 'PENDING',
                        started_at: null,
                        metadata: {
                            project: { id: 'project-a' },
                            model: { id: 'q-model', name: 'q model', architecture: ARCH_ID },
                            model_variant: { id: 'variant-1' },
                            max_calibration_subset_size: 100,
                        },
                    }),
                ])
            )
        );

        renderDialog();

        await screen.findByText('q model');
        expect(screen.getByText('—')).toBeInTheDocument();
    });
});
