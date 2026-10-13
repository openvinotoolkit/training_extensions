// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';
import { vi } from 'vitest';

import { getMockedTrainJob } from '../../../mocks/mock-job';
import { http } from '../../api/utils';
import { paths } from '../../constants/paths';
import { server } from '../../msw-node-setup';
import { createQueryClient } from '../../query-client/query-client';
import { MockEventSourceConstructor, resetMockEventSource } from '../../test-utils/mock-event-source';
import { ActiveJobsSync } from './active-jobs-sync.component';

describe('ActiveJobsSync', () => {
    beforeEach(() => {
        resetMockEventSource();
    });

    it('opens an SSE stream for every active train/quantize job and none for terminal ones', async () => {
        const runningJob = getMockedTrainJob({ job_id: 'running-job', status: 'RUNNING' });
        const doneJob = getMockedTrainJob({ job_id: 'done-job', status: 'DONE' });
        server.use(http.get('/api/jobs', () => HttpResponse.json([runningJob, doneJob])));

        render(<ActiveJobsSync />);

        await waitFor(() => {
            expect(MockEventSourceConstructor).toHaveBeenCalledTimes(1);
            expect(MockEventSourceConstructor).toHaveBeenCalledWith(expect.stringContaining(runningJob.job_id));
        });
    });

    it('closes the stream once the job reaches a terminal status', async () => {
        const runningJob = getMockedTrainJob({ job_id: 'running-job', status: 'RUNNING' });
        server.use(http.get('/api/jobs', () => HttpResponse.json([runningJob])));

        render(<ActiveJobsSync />);

        await waitFor(() => expect(MockEventSourceConstructor).toHaveBeenCalledTimes(1));

        const eventSource = MockEventSourceConstructor.mock.results.at(-1)?.value;
        const closeSpy = vi.spyOn(eventSource, 'close');

        eventSource.onmessage?.({
            data: JSON.stringify({ ...runningJob, status: 'DONE' }),
        } as unknown as Event);

        await waitFor(() => expect(closeSpy).toHaveBeenCalled());
    });

    it('does not crash on the project-list route, which has no projectId in the URL', async () => {
        const runningJob = getMockedTrainJob({ job_id: 'running-job', status: 'RUNNING' });
        server.use(http.get('/api/jobs', () => HttpResponse.json([runningJob])));

        render(<ActiveJobsSync />, {
            route: paths.project.index({}),
            path: paths.project.index.pattern,
        });

        await waitFor(() => expect(MockEventSourceConstructor).toHaveBeenCalledTimes(1));
    });

    it("invalidates the job's own project models cache, not the currently viewed project's", async () => {
        const otherProjectJob = getMockedTrainJob({
            job_id: 'other-project-job',
            status: 'RUNNING',
            metadata: { ...getMockedTrainJob().metadata, project: { id: 'other-project' } },
        });
        server.use(http.get('/api/jobs', () => HttpResponse.json([otherProjectJob])));

        const queryClient = createQueryClient();
        const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

        render(<ActiveJobsSync />, { queryClient, path: paths.project.details({ projectId: '123' }) });

        await waitFor(() => expect(MockEventSourceConstructor).toHaveBeenCalledTimes(1));

        const eventSource = MockEventSourceConstructor.mock.results.at(-1)?.value;
        eventSource.onmessage?.({
            data: JSON.stringify({ ...otherProjectJob, status: 'DONE' }),
        } as unknown as Event);

        await waitFor(() => {
            expect(invalidateSpy).toHaveBeenCalledWith(
                expect.objectContaining({
                    queryKey: expect.arrayContaining([
                        expect.objectContaining({ params: { path: { project_id: 'other-project' } } }),
                    ]),
                })
            );
        });
        expect(invalidateSpy).not.toHaveBeenCalledWith(
            expect.objectContaining({
                queryKey: expect.arrayContaining([
                    expect.objectContaining({ params: { path: { project_id: '123' } } }),
                ]),
            })
        );
    });
});
