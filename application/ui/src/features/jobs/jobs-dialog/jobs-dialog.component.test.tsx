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
import { server } from '../../../msw-node-setup';
import { ALL_PROJECTS, type ModelJob } from '../utils';
import { JobsDialog } from './jobs-dialog.component';

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
    render(<JobsDialog initialProjectId={ALL_PROJECTS} onClose={vi.fn()} {...props} />);

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

    it('shows the row status badge only on the All tab', async () => {
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

    it('shows the project name in the meta line only in All projects scope, and "Waiting to start" for scheduled jobs', async () => {
        mockJobs([trainJob({ job_id: 'pending', status: 'PENDING', started_at: null })]);

        const { unmount } = renderDialog({ initialProjectId: ALL_PROJECTS });

        expect(await screen.findByText('Alpha · Waiting to start...')).toBeInTheDocument();
        unmount();

        renderDialog({ initialProjectId: 'project-a', currentProjectId: 'project-a' });

        expect(await screen.findByText('Waiting to start...')).toBeInTheDocument();
    });

    it('shows the running percentage in the meta line and a bottom progress bar, only for running jobs', async () => {
        mockJobs([
            trainJob({ job_id: 'running', status: 'RUNNING', progress: 42, started_at: '2026-01-01T00:00:00Z' }),
        ]);

        renderDialog();

        await screen.findByText('running model');
        expect(screen.getByText(/42%/)).toBeInTheDocument();
        expect(document.body.querySelector('[style*="background-color: var(--energy-blue)"]')).toBeInTheDocument();
    });

    it('does not render a progress bar for a non-running job', async () => {
        mockJobs([trainJob({ job_id: 'pending', status: 'PENDING', started_at: null })]);

        renderDialog();

        await screen.findByText('pending model');
        expect(document.body.querySelector('[style*="background-color: var(--energy-blue)"]')).not.toBeInTheDocument();
    });

    it('shows the error message in red below the meta line for a failed job', async () => {
        mockJobs([
            trainJob({
                job_id: 'failed',
                status: 'FAILED',
                started_at: '2026-01-01T00:00:00Z',
                finished_at: '2026-01-01T00:05:00Z',
                error: 'CUDA out of memory',
            }),
        ]);

        renderDialog();

        expect(await screen.findByText('CUDA out of memory')).toBeInTheDocument();
    });

    it('shows "Cancel" only for pending and running jobs, not for cancelling/done/failed/cancelled', async () => {
        mockJobs([trainJob({ job_id: 'pending', status: 'PENDING', started_at: null })]);
        renderDialog();
        await userEvent.click(await screen.findByRole('button', { name: 'Job actions for pending model' }));
        expect(screen.getByRole('menuitem', { name: 'View logs' })).toBeInTheDocument();
        expect(screen.getByRole('menuitem', { name: 'Cancel' })).toBeInTheDocument();
    });

    it.each(['CANCELLING', 'DONE', 'FAILED', 'CANCELLED'])('hides "Cancel" for a %s job', async (status) => {
        mockJobs([
            trainJob({
                job_id: 'job',
                status,
                started_at: '2026-01-01T00:00:00Z',
                finished_at: status === 'CANCELLING' ? null : '2026-01-01T00:05:00Z',
            }),
        ]);
        renderDialog();
        await userEvent.click(await screen.findByRole('button', { name: 'Job actions for job model' }));
        expect(screen.getByRole('menuitem', { name: 'View logs' })).toBeInTheDocument();
        expect(screen.queryByRole('menuitem', { name: 'Cancel' })).not.toBeInTheDocument();
    });

    it('confirms and cancels a job through the actions menu', async () => {
        const cancelSpy = vi.fn();
        mockJobs([trainJob({ job_id: 'running', status: 'RUNNING', started_at: '2026-01-01T00:00:00Z' })]);
        server.use(
            http.post('/api/jobs/{job_id}:cancel', () => {
                cancelSpy();
                return HttpResponse.json(null, { status: 204 });
            })
        );

        renderDialog();

        await userEvent.click(await screen.findByRole('button', { name: 'Job actions for running model' }));
        await userEvent.click(screen.getByRole('menuitem', { name: 'Cancel' }));

        expect(await screen.findByRole('alertdialog', { name: 'Cancel job' })).toBeInTheDocument();

        await userEvent.click(screen.getByRole('button', { name: 'Cancel job' }));

        await vi.waitFor(() => expect(cancelSpy).toHaveBeenCalled());
    });

    it('opens the full-screen logs dialog on "View logs" and returns to the jobs dialog on close, tab unchanged', async () => {
        mockJobs([trainJob({ job_id: 'running', status: 'RUNNING', started_at: '2026-01-01T00:00:00Z' })]);
        renderDialog();

        await userEvent.click(await screen.findByRole('tab', { name: /^Running\b/ }));
        await userEvent.click(screen.getByRole('button', { name: 'Job actions for running model' }));
        await userEvent.click(screen.getByRole('menuitem', { name: 'View logs' }));

        expect(await screen.findByText('Training Logs')).toBeInTheDocument();

        await userEvent.click(screen.getByRole('button', { name: 'Close dialog' }));

        await waitFor(() => expect(screen.queryByText('Training Logs')).not.toBeInTheDocument());
        expect(screen.getByRole('button', { name: /Filter jobs by project/ })).toHaveTextContent('All projects');
        expect(screen.getByRole('tab', { name: /^Running\b/, selected: true })).toBeInTheDocument();
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
