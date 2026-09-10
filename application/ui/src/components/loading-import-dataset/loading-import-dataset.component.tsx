// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { useImportJobStatus } from 'hooks/api/jobs/use-import-job-status.hook';
import { useDeleteStagedDataset } from 'hooks/api/staged-dataset.hook';
import { isInvalidJob, isJobFailed, isJobPending, isJobRunning } from 'hooks/api/util';

import { formatBytes } from '../../shared/util';
import { ImportActiveJob } from '../import-card-status/import-active-job/import-active-job.component';
import { ImportFailedJob } from '../import-card-status/import-failed-job/import-failed-job.component';
import { toast } from '../toast/toast.component';

type LoadingImportDatasetProps = {
    jobId: string;
    size: number;
    fileName: string;
    stagedDatasetId: string;
    onSuccess: () => Promise<void> | void;
    deleteEntry: () => void;
};

export const LoadingImportDataset = ({
    size,
    fileName,
    jobId,
    stagedDatasetId,
    onSuccess,
    deleteEntry,
}: LoadingImportDatasetProps) => {
    const { t } = useTranslation();
    const deleteStagedFileMutation = useDeleteStagedDataset({ stagedDatasetId });

    const { data: job, ...response } = useImportJobStatus({
        jobId,
        onSuccess: async () => {
            deleteEntry();
            deleteStagedFileMutation.mutate();
            await onSuccess();

            toast({
                message: t('dataset.import.importedSuccess', { fileName, size: formatBytes(size) }),
                type: 'success',
            });
        },
        onError: (error) => {
            if (isInvalidJob(error)) {
                deleteEntry();
                deleteStagedFileMutation.mutate();
            }
        },
    });

    const isRunningOrPending = job !== undefined && (isJobRunning(job) || isJobPending(job));

    return (
        <>
            {isJobFailed(job) && (
                <ImportFailedJob
                    fileName={fileName}
                    size={size}
                    error={job?.error ?? ''}
                    message={job?.message ?? ''}
                    stagedDatasetId={stagedDatasetId}
                    deleteEntry={deleteEntry}
                />
            )}

            {response.isError && (
                <ImportFailedJob
                    size={size}
                    fileName={fileName}
                    error={`${response.error?.detail ?? t('dataset.import.unknownErrorDetail')}`}
                    message={t('dataset.import.genericError')}
                    stagedDatasetId={stagedDatasetId}
                    deleteEntry={deleteEntry}
                />
            )}

            {isRunningOrPending && (
                <ImportActiveJob
                    job={job}
                    size={size}
                    fileName={fileName}
                    stagedDatasetId={stagedDatasetId}
                    deleteEntry={deleteEntry}
                />
            )}
        </>
    );
};
