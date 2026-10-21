// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { QuantizeJob, TrainJob } from '@/api/types';

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

    const finishedAt = job.finished_at ? new Date(job.finished_at).getTime() : Date.now();
    const elapsedSeconds = Math.floor(Math.max(0, finishedAt - new Date(job.started_at).getTime()) / 1000);
    const hours = Math.floor(elapsedSeconds / 3600);
    const minutes = Math.floor((elapsedSeconds % 3600) / 60);
    const seconds = elapsedSeconds % 60;

    if (hours >= 1) return `${hours}h ${minutes}m ${seconds}s`;
    if (minutes >= 1) return `${minutes}m ${seconds}s`;
    return `${seconds}s`;
};
