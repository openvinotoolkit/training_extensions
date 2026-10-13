// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedQuantizeJob, getMockedTrainJob } from 'mocks/mock-job';

import {
    ALL_PROJECTS,
    countByStatusGroup,
    filterJobs,
    formatElapsed,
    getStatusGroup,
    sortJobs,
    type ModelJob,
} from './utils';

describe('getStatusGroup', () => {
    it.each([
        ['PENDING', 'scheduled'],
        ['RUNNING', 'running'],
        ['CANCELLING', 'running'],
        ['DONE', 'finished'],
        ['FAILED', 'failed'],
        ['CANCELLED', 'cancelled'],
    ])('maps %s to %s', (status, group) => {
        expect(getStatusGroup(status)).toBe(group);
    });
});

describe('countByStatusGroup', () => {
    it('counts jobs per group and totals them under "all"', () => {
        const jobs: ModelJob[] = [
            getMockedTrainJob({ job_id: '1', status: 'RUNNING' }),
            getMockedTrainJob({ job_id: '2', status: 'CANCELLING' }),
            getMockedTrainJob({ job_id: '3', status: 'PENDING' }),
            getMockedQuantizeJob({ job_id: '4', status: 'DONE' }),
            getMockedTrainJob({ job_id: '5', status: 'FAILED' }),
            getMockedTrainJob({ job_id: '6', status: 'CANCELLED' }),
        ];

        expect(countByStatusGroup(jobs)).toEqual({
            all: 6,
            running: 2,
            scheduled: 1,
            finished: 1,
            failed: 1,
            cancelled: 1,
        });
    });

    it('returns all zeros (and all: 0) for an empty list', () => {
        expect(countByStatusGroup([])).toEqual({
            all: 0,
            running: 0,
            scheduled: 0,
            finished: 0,
            failed: 0,
            cancelled: 0,
        });
    });
});

describe('filterJobs', () => {
    const jobA = getMockedTrainJob({
        job_id: 'a',
        status: 'RUNNING',
        metadata: { ...getMockedTrainJob().metadata, project: { id: 'project-1' } },
    });
    const jobB = getMockedTrainJob({
        job_id: 'b',
        status: 'DONE',
        metadata: { ...getMockedTrainJob().metadata, project: { id: 'project-2' } },
    });
    const jobs = [jobA, jobB];

    it('filters by project id', () => {
        expect(filterJobs(jobs, 'project-1', 'all')).toEqual([jobA]);
    });

    it('keeps every project when scoped to ALL_PROJECTS', () => {
        expect(filterJobs(jobs, ALL_PROJECTS, 'all')).toEqual(jobs);
    });

    it('filters by status group', () => {
        expect(filterJobs(jobs, ALL_PROJECTS, 'finished')).toEqual([jobB]);
    });

    it('keeps every status when group is "all"', () => {
        expect(filterJobs(jobs, ALL_PROJECTS, 'all')).toEqual(jobs);
    });

    it('combines project and status group filters', () => {
        expect(filterJobs(jobs, 'project-2', 'finished')).toEqual([jobB]);
        expect(filterJobs(jobs, 'project-2', 'running')).toEqual([]);
    });
});

describe('sortJobs', () => {
    it('puts jobs that have not started first, then orders by started_at descending', () => {
        const notStarted = getMockedTrainJob({ job_id: 'not-started', started_at: null });
        const older = getMockedTrainJob({ job_id: 'older', started_at: '2026-01-01T00:00:00.000000+00:00' });
        const newer = getMockedTrainJob({ job_id: 'newer', started_at: '2026-01-03T00:00:00.000000+00:00' });

        expect(sortJobs([older, notStarted, newer]).map((job) => job.job_id)).toEqual([
            'not-started',
            'newer',
            'older',
        ]);
    });
});

describe('formatElapsed', () => {
    beforeEach(() => {
        vi.useFakeTimers();
        vi.setSystemTime(new Date('2026-01-01T01:00:00.000000+00:00'));
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    it('returns "—" for a job that has not started', () => {
        expect(formatElapsed(getMockedTrainJob({ started_at: null }))).toBe('—');
    });

    it('formats hours, minutes and seconds between start and finish', () => {
        const job = getMockedTrainJob({
            started_at: '2026-01-01T00:00:00.000000+00:00',
            finished_at: '2026-01-01T01:02:03.000000+00:00',
        });

        expect(formatElapsed(job)).toBe('1h 2m 3s');
    });

    it('elapses to now when the job has not finished yet', () => {
        const job = getMockedTrainJob({
            started_at: '2026-01-01T00:59:30.000000+00:00',
            finished_at: null,
        });

        expect(formatElapsed(job)).toBe('30s');
    });
});
