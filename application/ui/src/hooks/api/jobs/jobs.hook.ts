// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useRef } from 'react';

import { $api } from '@/api';
import type { Job, QuantizeJob, TrainJob } from '@/api/types';
import { toast } from '@/components/toast/toast.component';
import { useQueryClient } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { getQueryKey } from '../../../query-client/query-client';
import { useDismissedJobs } from '../../storage/use-dismissed-jobs.hook';
import { useSSE } from '../../use-sse.hook';
import { isJobFailed, isQuantizeJob, isTrainJob } from '../util';

const TERMINAL_STATUSES: string[] = ['DONE', 'FAILED', 'CANCELLED'];
const ERROR_MESSAGE = 'Job failed. Please check the logs for details and try again.';

export const useStreamJobStatus = (jobId: string | undefined) => {
    const queryClient = useQueryClient();
    const projectId = useProjectIdentifier();
    const modelIdRef = useRef<string | null>(null);

    const { close } = useSSE<Job>(jobId ? `/api/jobs/${jobId}/status` : undefined, {
        retry: true,
        onMessage: (updatedJob) => {
            if (isQuantizeJob(updatedJob)) {
                modelIdRef.current = updatedJob.metadata.model.id;
            }
            // Update the job in the cache optimistically to reflect real-time progress
            queryClient.setQueryData<Job[]>(['get', '/api/jobs'], (prevJobs) => {
                if (!prevJobs) {
                    return [updatedJob];
                }

                return prevJobs.map((job) => (job.job_id === updatedJob.job_id ? updatedJob : job));
            });

            if (updatedJob.status === 'FAILED') {
                toast({
                    message: ERROR_MESSAGE,
                    type: 'error',
                });
            }

            if (TERMINAL_STATUSES.includes(updatedJob.status)) {
                close();
            }
        },
        onClose: () => {
            queryClient.invalidateQueries({ queryKey: getQueryKey(['get', '/api/jobs']) });
            queryClient.invalidateQueries({
                queryKey: getQueryKey([
                    'get',
                    '/api/projects/{project_id}/models',
                    { params: { path: { project_id: projectId } } },
                ]),
            });

            modelIdRef.current !== null &&
                queryClient.invalidateQueries({
                    queryKey: getQueryKey([
                        'get',
                        '/api/projects/{project_id}/models/{model_id}',
                        {
                            params: {
                                path: {
                                    project_id: projectId,
                                    model_id: modelIdRef.current,
                                },
                            },
                        },
                    ]),
                });
            modelIdRef.current = null;
        },
    });
};

// Same stream as useStreamJobStatus, but writes to the single-job cache instead of the job-list
// cache, so consumers that query a job by id (import/export) don't have to poll.
export const useStreamJobDetail = (jobId: string | undefined | null) => {
    const queryClient = useQueryClient();

    const { close } = useSSE<Job>(jobId ? `/api/jobs/${jobId}/status` : undefined, {
        retry: true,
        onMessage: (updatedJob) => {
            queryClient.setQueryData<Job>(
                getQueryKey(['get', '/api/jobs/{job_id}', { params: { path: { job_id: updatedJob.job_id } } }]),
                updatedJob
            );

            if (TERMINAL_STATUSES.includes(updatedJob.status)) {
                close();
            }
        },
        onClose: () => {
            if (!jobId) {
                return;
            }

            const jobKey = getQueryKey(['get', '/api/jobs/{job_id}', { params: { path: { job_id: jobId } } }]);
            const cachedJob = queryClient.getQueryData<Job>(jobKey);

            // The stream gave up before the job settled, so whatever is cached is stale
            if (!cachedJob || !TERMINAL_STATUSES.includes(cachedJob.status)) {
                queryClient.invalidateQueries({ queryKey: jobKey });
            }
        },
    });
};

export const useSubmitJob = () => {
    return $api.useMutation('post', '/api/jobs', {
        meta: {
            invalidateQueries: [['get', '/api/jobs']],
        },
    });
};

const useListJobs = () => {
    return $api.useQuery('get', '/api/jobs');
};

const isTrainOrQuantizeJob = (job: Job): job is TrainJob | QuantizeJob => isTrainJob(job) || isQuantizeJob(job);

export const useGetCurrentRunningJobs = (): (QuantizeJob | TrainJob)[] | undefined => {
    const projectId = useProjectIdentifier();
    const activeJobs = useListJobs();
    const { isJobDismissed } = useDismissedJobs();

    return activeJobs.data?.filter((job): job is TrainJob | QuantizeJob => {
        if (!isTrainOrQuantizeJob(job) || job.metadata.project.id !== projectId) {
            return false;
        }

        if (job.status === 'RUNNING' || job.status === 'PENDING') {
            return true;
        }

        // A failed train job is already surfaced as a "Failed" model, the other job types are not,
        // so they stay listed until the user dismisses them
        return isJobFailed(job) && !isTrainJob(job) && !isJobDismissed(job.job_id);
    });
};

export const useCancelJob = () => {
    return $api.useMutation('post', '/api/jobs/{job_id}:cancel', {
        meta: {
            invalidateQueries: [['get', '/api/jobs']],
        },
    });
};
