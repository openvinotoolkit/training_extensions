// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import type { Job, QuantizeJob, TrainJob } from '@/api/types';
import { useTranslation } from '@/i18n';
import { AlertDialog, Badge, Button, DialogContainer, Flex, Loading, Text } from '@geti-ui/ui';
import { useStreamJobStatus } from 'hooks/api/jobs/jobs.hook';
import { capitalize } from 'lodash-es';

import { JobRow, type JobRowColumnsProps } from './job-row.component';

import classes from './current-running-jobs.module.scss';

type RunningJobRowProps = JobRowColumnsProps & {
    job: TrainJob | QuantizeJob;
    onCancel?: () => void;
};

const StatusBadge = ({ status }: { status: string }) => {
    return (
        <Badge variant={'neutral'} UNSAFE_className={classes.runningStatusBadge}>
            <Text>
                <Flex alignItems={'center'} gap={'size-50'}>
                    <Loading size={'S'} mode={'inline'} />
                    {status}
                </Flex>
            </Text>
        </Badge>
    );
};

const StatusBadgeMessage = ({ status }: { status: string }) => {
    return (
        <Badge variant={'neutral'} UNSAFE_className={classes.statusBadge}>
            {status}
        </Badge>
    );
};

type CancelRunningJobProps = {
    job: Job;
    onCancel: () => void;
};

const CancelRunningJob = ({ job, onCancel }: CancelRunningJobProps) => {
    const { t } = useTranslation();
    const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState<boolean>(false);

    return (
        <>
            <Button
                isDisabled={job.status !== 'RUNNING' && job.status !== 'PENDING'}
                variant={'negative'}
                onPress={() => setIsDeleteDialogOpen(true)}
                aria-label={'Cancel job'}
            >
                {t('models.jobs.cancel')}
            </Button>
            <DialogContainer onDismiss={() => setIsDeleteDialogOpen(false)}>
                {isDeleteDialogOpen && (
                    <AlertDialog
                        title={t('models.jobs.stopDialog.title')}
                        variant='destructive'
                        primaryActionLabel={t('models.jobs.cancel')}
                        onPrimaryAction={onCancel}
                        cancelLabel={t('models.jobs.stopDialog.close')}
                    >
                        {t('models.jobs.stopDialog.confirmation')}
                    </AlertDialog>
                )}
            </DialogContainer>
        </>
    );
};

export const RunningJobRow = ({ job, onCancel, datasetRevisions, groupBy, modelArchitectures }: RunningJobRowProps) => {
    const { t } = useTranslation();
    useStreamJobStatus(job.job_id);

    const statusMessage =
        job.message || (job.status === 'PENDING' ? t('models.jobs.pending') : t('models.jobs.running'));
    const showStatusTagMessage =
        job.status.toLocaleLowerCase() !== statusMessage.replace('...', '').toLocaleLowerCase();

    return (
        <JobRow
            job={job}
            progress={job.progress}
            statusBadges={
                <>
                    <StatusBadge status={capitalize(job.status)} />
                    {showStatusTagMessage && <StatusBadgeMessage status={statusMessage} />}
                </>
            }
            actions={onCancel && <CancelRunningJob onCancel={onCancel} job={job} />}
            groupBy={groupBy}
            datasetRevisions={datasetRevisions}
            modelArchitectures={modelArchitectures}
        />
    );
};
