// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState, type ReactNode } from 'react';

import type { DatasetRevision, ModelArchitectureWithPerformanceCategory, QuantizeJob, TrainJob } from '@/api/types';
import { BottomProgressBar } from '@/components/bottom-progress-bar/bottom-progress-bar.component';
import { TrainingLogsDialog } from '@/components/training-logs/training-logs-dialog.component';
import { useTranslation } from '@/i18n';
import { Button, DialogContainer, Flex, Grid, Text } from '@geti-ui/ui';
import { isJobPending, isJobRunning, isTrainJob } from 'hooks/api/util';
import { DATE_TIME_FORMAT, useFormatDate } from 'hooks/use-format-date.hook';

import { useGetModel } from '../../hooks/api/use-get-model.hook';
import { ArchitectureColumn } from '../components/model-row/architecture-column.component';
import { DatasetColumn } from '../components/model-row/dataset-revision-column.component';
import { RUNNING_JOB_GRID_COLUMNS } from './running-job-table-header.component';

import classes from './current-running-jobs.module.scss';

export type JobRowColumnsProps = {
    datasetRevisions: DatasetRevision[];
    modelArchitectures: ModelArchitectureWithPerformanceCategory[];
};

type JobRowProps = JobRowColumnsProps & {
    job: TrainJob | QuantizeJob;
    progress: number;
    statusBadges: ReactNode;
    actions?: ReactNode;
};

const ViewLogsButton = ({ job }: { job: TrainJob | QuantizeJob }) => {
    const { t } = useTranslation();
    const [isLogsDialogOpen, setIsLogsDialogOpen] = useState(false);

    return (
        <>
            <Button variant={'secondary'} onPress={() => setIsLogsDialogOpen(true)} aria-label={'View logs'}>
                {t('models.jobs.logs')}
            </Button>
            <DialogContainer type={'fullscreen'} onDismiss={() => setIsLogsDialogOpen(false)}>
                {isLogsDialogOpen && (
                    <TrainingLogsDialog
                        jobId={job.job_id}
                        projectId={job.metadata.project.id}
                        isJobActive={isJobRunning(job) || isJobPending(job)}
                    />
                )}
            </DialogContainer>
        </>
    );
};

export const JobRow = ({ job, progress, statusBadges, actions, datasetRevisions, modelArchitectures }: JobRowProps) => {
    const { t } = useTranslation();
    const modelId = job.metadata.model.id;
    const { data: trainingModel } = useGetModel(modelId, !isJobPending(job));

    const device = isTrainJob(job) ? job.metadata.device.name : null;

    const modelArchitectureId = job.metadata.model.architecture;
    const modelName = trainingModel?.name || job.metadata.model.name;

    const modelArchitecture = modelArchitectures.find(({ id }) => id === modelArchitectureId);

    const datasetRevision = datasetRevisions.find(({ id }) => id === trainingModel?.training_info.dataset_revision_id);
    const datasetViewName = isTrainJob(job) ? (job.metadata.model.dataset_view_name ?? undefined) : undefined;
    const labelSchemaRevision = trainingModel?.training_info.label_schema_revision ?? {};
    const labelsCount =
        'labels' in labelSchemaRevision && Array.isArray(labelSchemaRevision.labels)
            ? labelSchemaRevision.labels.length
            : undefined;

    const formatDate = useFormatDate(DATE_TIME_FORMAT);
    const formattedStartedAt = job.started_at ? (formatDate(job.started_at) ?? '-') : t('models.jobs.waitingToStart');

    return (
        <BottomProgressBar progress={progress}>
            <Grid
                columns={RUNNING_JOB_GRID_COLUMNS}
                alignItems={'center'}
                width={'100%'}
                columnGap={'size-200'}
                UNSAFE_className={classes.grid}
            >
                <Flex direction={'column'} justifyContent={'center'} gap={'size-50'}>
                    <Flex alignItems={'center'}>
                        <Text UNSAFE_className={classes.modelName}>{modelName}</Text>
                    </Flex>

                    <Flex alignItems={'start'} gap={'size-100'}>
                        {statusBadges}
                    </Flex>

                    <Text UNSAFE_className={classes.metaText}>
                        {t('models.jobs.startedAt', { time: formattedStartedAt })}
                    </Text>
                    {device && <Text UNSAFE_className={classes.metaText}>{t('models.jobs.device', { device })}</Text>}
                </Flex>

                <Flex alignItems={'start'} direction={'column'} gap={'size-100'}>
                    <ArchitectureColumn architectureId={modelArchitectureId} architecture={modelArchitecture} />
                </Flex>

                <Flex alignItems={'start'} direction={'column'} gap={'size-100'}>
                    <DatasetColumn
                        datasetRevision={datasetRevision}
                        labelsCount={labelsCount}
                        pendingDatasetName={datasetViewName}
                    />
                </Flex>

                <Flex gap={'size-100'} direction={'column'} alignItems={'center'}>
                    <ViewLogsButton job={job} />
                    {actions}
                </Flex>
            </Grid>
        </BottomProgressBar>
    );
};
