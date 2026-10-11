// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { BottomProgressBar } from '@/components/bottom-progress-bar/bottom-progress-bar.component';
import { useTranslation } from '@/i18n';
import { Flex, Grid, Loading, Text } from '@geti-ui/ui';
import { DATE_TIME_FORMAT, useFormatDate } from 'hooks/use-format-date.hook';

import { formatElapsed, getStatusGroup, type ModelJob } from '../utils';
import { JobActionsMenu } from './job-actions-menu.component';
import { JobStatusBadge } from './job-status-badge.component';

import classes from './jobs-dialog.module.scss';

export const JOB_GRID_COLUMNS = ['minmax(0, 1fr)', 'size-1600', 'size-2400', 'size-2000', 'size-500'];

const JOB_TYPE_KEY = {
    train: 'common.labels.training',
    quantize: 'jobs.type.quantize',
} as const satisfies Record<ModelJob['job_type'], string>;

type JobRowProps = {
    job: ModelJob;
    showStatusBadge: boolean;
    projectName?: string;
    architectureName: string;
    onViewLogs: () => void;
};

export const JobRow = ({ job, showStatusBadge, projectName, architectureName, onViewLogs }: JobRowProps) => {
    const { t } = useTranslation();
    const formatDateTime = useFormatDate(DATE_TIME_FORMAT);
    const isRunning = getStatusGroup(job.status) === 'running' && job.progress > 0;
    const device = 'device' in job.metadata ? job.metadata.device.name : '—';

    const meta = job.started_at
        ? [
              projectName,
              t('jobs.meta.started', { time: formatDateTime(job.started_at) }),
              t('jobs.meta.elapsed', { time: formatElapsed(job) }),
              isRunning && `${Math.round(job.progress)}%`,
          ]
        : [projectName, t('models.jobs.waitingToStart')];

    const grid = (
        <Grid columns={JOB_GRID_COLUMNS} columnGap={'size-200'} alignItems={'center'} UNSAFE_className={classes.grid}>
            <Flex direction={'column'} gap={'size-50'} minWidth={0}>
                <Flex gap={'size-100'} alignItems={'center'} wrap height={'size-600'}>
                    <Text UNSAFE_className={classes.modelName}>{job.metadata.model.name}</Text>
                    {showStatusBadge && <JobStatusBadge status={job.status} />}
                </Flex>
                <div className={classes.metaRow}>
                    <Text UNSAFE_className={classes.metaText}>{meta.filter(Boolean).join(' · ')}</Text>
                    {job.progress === 0 && <Loading size={'S'} mode={'inline'} />}
                </div>
                {job.message && <Text UNSAFE_className={classes.messageText}>{job.message}</Text>}
                {job.error && <Text UNSAFE_className={classes.errorText}>{job.error}</Text>}
            </Flex>
            <Text>{t(JOB_TYPE_KEY[job.job_type])}</Text>
            <Text>{architectureName}</Text>
            <Text>{device}</Text>
            <div>
                <JobActionsMenu job={job} onViewLogs={onViewLogs} />
            </div>
        </Grid>
    );

    return isRunning ? (
        <BottomProgressBar progress={job.progress} color='var(--energy-blue)'>
            {grid}
        </BottomProgressBar>
    ) : (
        grid
    );
};
