// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Job } from '@/api/types';
import { useTranslation } from '@/i18n';
import { Loading } from '@geti-ui/ui';
import { useDeleteStagedDataset } from 'hooks/api/staged-dataset.hook';
import { getJobProgress, isJobRunning } from 'hooks/api/util';
import capitalize from 'lodash-es/capitalize';

import { CancelJobConfirmation } from '../../../features/dataset/import-export/cancel-job-confirmation/cancel-job-confirmation.component';
import { BottomProgressBar } from '../../../features/models/model-listing/current-running-jobs/bottom-progress-bar.component';
import { formatBytes } from '../../../shared/util';
import { JobStatusCard } from '../../job-status-card/job-status-card.component';

type ImportActiveJobProps = {
    job: Job;
    size: number;
    fileName: string;
    stagedDatasetId: string;
    deleteEntry: () => void;
};

export const ImportActiveJob = ({ job, size, fileName, stagedDatasetId, deleteEntry }: ImportActiveJobProps) => {
    const { t } = useTranslation();
    const deleteFileMutation = useDeleteStagedDataset({ stagedDatasetId, deleteEntry });

    const isRunning = isJobRunning(job);
    const progress = getJobProgress(job?.progress);

    const handleRemove = () => {
        return deleteFileMutation.mutateAsync();
    };

    return (
        <BottomProgressBar progress={progress}>
            <JobStatusCard
                title={t('dataset.import.jobTitle', { fileName, size: formatBytes(size) })}
                actionButtons={<CancelJobConfirmation jobId={job.job_id} onRemove={handleRemove} />}
                message={t('dataset.import.processingMessage', { fileName })}
                bottomIcon={<Loading mode='inline' size='S' />}
                bottomIconMessage={job?.message ?? capitalize(job.status.toLocaleLowerCase())}
                bottomRightMessage={isRunning ? `${progress}%` : undefined}
            />
        </BottomProgressBar>
    );
};
