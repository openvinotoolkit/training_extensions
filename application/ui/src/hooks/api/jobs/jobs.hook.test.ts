// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import * as toastModule from '@/components/toast/toast.component';
import { act, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { renderHook } from 'test-utils/render';
import { vi } from 'vitest';

import { getMockedJob, getMockedQuantizeJob } from '../../../../mocks/mock-job';
import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { MockEventSourceConstructor, resetMockEventSource } from '../../../test-utils/mock-event-source';
import { useDismissedJobs } from '../../storage/use-dismissed-jobs.hook';
import { useGetCurrentRunningJobs, useStreamJobStatus } from './jobs.hook';

const PROJECT_ID = '123';

const createMockJobForProject = (overrides: Partial<ReturnType<typeof getMockedJob>> = {}) =>
    getMockedJob({
        metadata: {
            project: { id: PROJECT_ID },
            model: {
                id: 'ef3983f1-cef0-4ebe-91db-7330f1dd6e27',
                name: 'SSD (ef3983f1)',
                architecture: 'SSD',
                dataset_revision_id: 'ds-rev-1',
            },
            device: {
                type: 'cpu',
                name: 'CPU',
            },
        },
        ...overrides,
    });

describe('useGetCurrentRunningJobs', () => {
    beforeEach(() => {
        resetMockEventSource();
    });

    afterEach(() => {
        sessionStorage.clear();
    });

    it('returns an empty array when there are no active running jobs', async () => {
        server.use(http.get('/api/jobs', () => HttpResponse.json([])));

        const { result } = renderHook(() => useGetCurrentRunningJobs());

        await waitFor(() => {
            expect(result.current).toEqual([]);
        });
    });

    it('returns the active running job for the current project', async () => {
        const job = createMockJobForProject();
        server.use(http.get('/api/jobs', () => HttpResponse.json([job])));

        const { result } = renderHook(() => useGetCurrentRunningJobs());

        await waitFor(() => {
            expect(result.current).toHaveLength(1);
            expect(result.current?.[0].job_id).toBe(job.job_id);
        });
    });

    it('ignores jobs from other projects', async () => {
        const otherProjectJob = getMockedJob({
            metadata: {
                project: { id: 'other-project' },
                model: {
                    id: 'ef3983f1-cef0-4ebe-91db-7330f1dd6e27',
                    name: 'SSD (ef3983f1)',
                    architecture: 'SSD',
                    dataset_revision_id: 'ds-rev-1',
                },
                device: {
                    type: 'cpu',
                    name: 'CPU',
                },
            },
        });
        server.use(http.get('/api/jobs', () => HttpResponse.json([otherProjectJob])));

        const { result } = renderHook(() => useGetCurrentRunningJobs());

        await waitFor(() => {
            expect(result.current).toEqual([]);
        });
    });

    it('does not subscribe to SSE when no active running job matches the project', async () => {
        server.use(http.get('/api/jobs', () => HttpResponse.json([])));

        renderHook(() => useGetCurrentRunningJobs());

        await waitFor(() => {
            expect(MockEventSourceConstructor).not.toHaveBeenCalled();
        });
    });

    it('keeps failed jobs that are not train jobs', async () => {
        const failedQuantizeJob = getMockedQuantizeJob({
            status: 'FAILED',
            metadata: {
                ...getMockedQuantizeJob().metadata,
                project: { id: PROJECT_ID },
            },
        });
        server.use(http.get('/api/jobs', () => HttpResponse.json([failedQuantizeJob])));

        const { result } = renderHook(() => useGetCurrentRunningJobs());

        await waitFor(() => {
            expect(result.current).toHaveLength(1);
            expect(result.current?.[0].job_id).toBe(failedQuantizeJob.job_id);
        });
    });

    it('ignores failed train jobs', async () => {
        const failedTrainJob = createMockJobForProject({ status: 'FAILED' });
        server.use(http.get('/api/jobs', () => HttpResponse.json([failedTrainJob])));

        const { result } = renderHook(() => useGetCurrentRunningJobs());

        await waitFor(() => {
            expect(result.current).toEqual([]);
        });
    });

    it('ignores failed jobs that have been dismissed', async () => {
        const failedQuantizeJob = getMockedQuantizeJob({
            status: 'FAILED',
            metadata: {
                ...getMockedQuantizeJob().metadata,
                project: { id: PROJECT_ID },
            },
        });
        server.use(http.get('/api/jobs', () => HttpResponse.json([failedQuantizeJob])));

        const { result } = renderHook(() => {
            const { dismissJob } = useDismissedJobs();

            return { jobs: useGetCurrentRunningJobs(), dismissJob };
        });

        await waitFor(() => {
            expect(result.current.jobs).toHaveLength(1);
        });

        act(() => {
            result.current.dismissJob(failedQuantizeJob.job_id);
        });

        await waitFor(() => {
            expect(result.current.jobs).toEqual([]);
        });
    });
});

describe('useStreamJobStatus', () => {
    beforeEach(() => {
        resetMockEventSource();
    });

    it('shows a toast when job status is FAILED', async () => {
        const toastSpy = vi.spyOn(toastModule, 'toast');
        const failedJob = getMockedJob({ status: 'FAILED' });

        renderHook(() => useStreamJobStatus(failedJob.job_id));

        await waitFor(() => {
            expect(MockEventSourceConstructor).toHaveBeenCalled();
        });
        const eventSource = MockEventSourceConstructor.mock.results.at(-1)?.value;

        expect(eventSource).toBeDefined();
        eventSource.onmessage?.({ data: JSON.stringify(failedJob) } as unknown as Event);

        await waitFor(() => {
            expect(toastSpy).toHaveBeenCalledWith(
                expect.objectContaining({
                    message: 'Job failed. Please check the logs for details and try again.',
                    type: 'error',
                })
            );
        });
    });
});
