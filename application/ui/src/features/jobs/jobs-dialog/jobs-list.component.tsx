// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { EmptyDataset } from '@/assets/illustrations';
import { useTranslation } from '@/i18n';
import { Grid, Heading, IllustratedMessage, Text } from '@geti-ui/ui';

import type { ModelJob, StatusGroup } from '../utils';
import { JOB_GRID_COLUMNS, JobRow } from './job-row.component';

import classes from './jobs-dialog.module.scss';

type JobsListProps = {
    jobs: ModelJob[];
    isAllProjects: boolean;
    showStatusBadge: boolean;
    statusGroup: StatusGroup;
    projectName: (projectId: string) => string;
    architectureName: (architectureId: string) => string;
    onViewLogs: (job: ModelJob) => void;
};

export const JobsList = ({
    jobs,
    isAllProjects,
    showStatusBadge,
    statusGroup,
    projectName,
    architectureName,
    onViewLogs,
}: JobsListProps) => {
    const { t } = useTranslation();
    const hasJobs = jobs.length > 0;

    return (
        <div aria-label='Jobs' className={classes.list}>
            {hasJobs && (
                <Grid
                    columns={JOB_GRID_COLUMNS}
                    columnGap={'size-200'}
                    UNSAFE_className={`${classes.grid} ${classes.header}`}
                >
                    <Text>{t('common.labels.name')}</Text>
                    <Text>{t('jobs.columns.type')}</Text>
                    <Text>{t('common.labels.architecture')}</Text>
                    <Text>{t('common.labels.device')}</Text>
                    <div />
                </Grid>
            )}
            {hasJobs ? (
                jobs.map((job) => (
                    <JobRow
                        key={job.job_id}
                        job={job}
                        showStatusBadge={showStatusBadge}
                        projectName={isAllProjects ? projectName(job.metadata.project.id) : undefined}
                        architectureName={architectureName(job.metadata.model.architecture)}
                        onViewLogs={() => onViewLogs(job)}
                    />
                ))
            ) : (
                <IllustratedMessage>
                    <EmptyDataset />
                    <Heading>{t(`jobs.emptyState.${statusGroup}`)}</Heading>
                </IllustratedMessage>
            )}
        </div>
    );
};
