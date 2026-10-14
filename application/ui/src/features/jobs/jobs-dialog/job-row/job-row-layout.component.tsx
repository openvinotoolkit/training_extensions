// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { ReactNode } from 'react';

import { useTranslation } from '@/i18n';
import { Flex, Grid, Text, View } from '@geti-ui/ui';
import { Link } from 'react-router';

import { paths } from '../../../../constants/paths';
import { type ModelJob } from '../../utils';
import { JobStatusBadge } from './job-status-badge.component';

import dialogClasses from '../jobs-dialog.module.scss';
import classes from './job-row.module.scss';

export const JOB_GRID_COLUMNS = ['minmax(0, 1fr)', 'size-1600', 'size-2400', 'size-2000', 'size-500'];

const JOB_TYPE_KEY = {
    train: 'common.labels.training',
    quantize: 'jobs.type.quantize',
} as const satisfies Record<ModelJob['job_type'], string>;

export type JobRowProps = {
    job: ModelJob;
    showStatusBadge: boolean;
    projectName?: string;
    architectureName: string;
    onViewLogs: () => void;
};

type JobRowLayoutProps = Pick<JobRowProps, 'job' | 'showStatusBadge' | 'projectName' | 'architectureName'> & {
    meta: ReactNode;
    spinner?: ReactNode;
    actions?: ReactNode;
};

export const JobRowLayout = ({
    job,
    showStatusBadge,
    projectName,
    architectureName,
    meta,
    spinner,
    actions,
}: JobRowLayoutProps) => {
    const { t } = useTranslation();
    const device = 'device' in job.metadata ? job.metadata.device.name : '—';

    return (
        <Grid
            columns={JOB_GRID_COLUMNS}
            columnGap={'size-200'}
            alignItems={'center'}
            UNSAFE_className={dialogClasses.grid}
        >
            <Flex direction={'column'} gap={'size-50'} minWidth={0}>
                <Flex gap={'size-100'} alignItems={'center'} wrap height={'size-600'}>
                    <Text UNSAFE_className={classes.modelName}>{job.metadata.model.name}</Text>
                    {showStatusBadge && <JobStatusBadge status={job.status} />}
                </Flex>
                <View UNSAFE_className={classes.metaRow}>
                    {projectName && (
                        <Link
                            to={paths.project.models({ projectId: job.metadata.project.id })}
                            className={classes.projectLink}
                        >
                            {projectName}
                        </Link>
                    )}
                    {projectName && ' · '}
                    <Text UNSAFE_className={classes.metaText}>{meta}</Text>
                    {spinner}
                </View>
                {job.message && <Text UNSAFE_className={classes.messageText}>{job.message}</Text>}
            </Flex>
            <Text>{t(JOB_TYPE_KEY[job.job_type])}</Text>
            <Text>{architectureName}</Text>
            <Text>{device}</Text>
            <View>{actions}</View>
        </Grid>
    );
};
