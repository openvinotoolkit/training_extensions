// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { QuantizeJob, TrainJob } from '@/api/types';
import dayjs from 'dayjs';
import durationPlugin from 'dayjs/plugin/duration';

dayjs.extend(durationPlugin);

export type ModelJob = TrainJob | QuantizeJob;

export type StatusGroup = 'all' | 'running' | 'finished' | 'scheduled' | 'cancelled' | 'failed';

export const ALL_PROJECTS = 'all';

export const getStatusGroup = (status: string): Exclude<StatusGroup, 'all'> => {
    switch (status) {
        case 'PENDING':
            return 'scheduled';
        case 'DONE':
            return 'finished';
        case 'CANCELLED':
            return 'cancelled';
        case 'FAILED':
            return 'failed';
        default:
            return 'running';
    }
};

export const countByStatusGroup = (jobs: ModelJob[]): Record<StatusGroup, number> => {
    const counts: Record<StatusGroup, number> = {
        all: jobs.length,
        running: 0,
        finished: 0,
        scheduled: 0,
        cancelled: 0,
        failed: 0,
    };

    jobs.forEach((job) => counts[getStatusGroup(job.status)]++);

    return counts;
};

export const filterJobs = (jobs: ModelJob[], projectId: string, group: StatusGroup): ModelJob[] =>
    jobs.filter(
        (job) =>
            (projectId === ALL_PROJECTS || job.metadata.project.id === projectId) &&
            (group === 'all' || getStatusGroup(job.status) === group)
    );

const startedAt = (job: ModelJob): number =>
    job.started_at ? new Date(job.started_at).getTime() : Number.MAX_SAFE_INTEGER;

export const sortJobs = (jobs: ModelJob[]): ModelJob[] => jobs.toSorted((a, b) => startedAt(b) - startedAt(a));

export const formatElapsed = (job: ModelJob): string => {
    if (!job.started_at) return '—';

    const elapsed = dayjs.duration(Math.max(0, dayjs(job.finished_at ?? undefined).diff(job.started_at)));
    const hours = Math.floor(elapsed.asHours());

    if (hours >= 1) return `${hours}h ${elapsed.format('m[m] s[s]')}`;
    if (elapsed.asMinutes() >= 1) return elapsed.format('m[m] s[s]');
    return elapsed.format('s[s]');
};
