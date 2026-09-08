// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState, type ReactNode } from 'react';

import type { DatasetRevision, ModelArchitectureWithPerformanceCategory, QuantizeJob, TrainJob } from '@/api/types';
import { Button, DialogContainer, Flex, Grid, Text } from '@geti-ui/ui';
import { isJobPending, isTrainJob } from 'hooks/api/util';

import { formatDateTime } from '../../../../shared/date-utils';
import { useGetModel } from '../../hooks/api/use-get-model.hook';
import { TrainingLogsDialog } from '../../training-logs/training-logs-dialog.component';
import { ArchitectureColumn } from '../components/model-row/architecture-column.component';
import { DatasetColumn } from '../components/model-row/dataset-revision-column.component';
import { GroupByMode } from '../types';
import { BottomProgressBar } from './bottom-progress-bar.component';
import { RUNNING_JOB_GRID_COLUMNS } from './running-job-table-header.component';

import classes from './current-running-jobs.module.scss';

export type JobRowColumnsProps = {
    groupBy: GroupByMode;
    datasetRevisions: DatasetRevision[];
    modelArchitectures: ModelArchitectureWithPerformanceCategory[];
};

type JobRowProps = JobRowColumnsProps & {
    job: TrainJob | QuantizeJob;
    progress: number;
    statusBadges: ReactNode;
    actions?: ReactNode;
};

const ViewLogsButton = ({ jobId }: { jobId: string }) => {
    const [isLogsDialogOpen, setIsLogsDialogOpen] = useState(false);

    return (
        <>
            <Button variant={'secondary'} onPress={() => setIsLogsDialogOpen(true)} aria-label={'View logs'}>
                Logs
            </Button>
            <DialogContainer type={'fullscreen'} onDismiss={() => setIsLogsDialogOpen(false)}>
                {isLogsDialogOpen && <TrainingLogsDialog jobId={jobId} />}
            </DialogContainer>
        </>
    );
};

export const JobRow = ({
    job,
    progress,
    statusBadges,
    actions,
    groupBy,
    datasetRevisions,
    modelArchitectures,
}: JobRowProps) => {
    const modelId = job.metadata.model.id;
    const { data: trainingModel } = useGetModel(modelId, !isJobPending(job));

    const device = isTrainJob(job) ? job.metadata.device.name : null;

    const modelArchitectureId = job.metadata.model.architecture;
    const modelName = trainingModel?.name || job.metadata.model.name;

    const modelArchitecture = modelArchitectures.find(({ id }) => id === modelArchitectureId);

    const datasetRevision = datasetRevisions.find(({ id }) => id === trainingModel?.training_info.dataset_revision_id);
    const labelSchemaRevision = trainingModel?.training_info.label_schema_revision ?? {};
    const labelsCount =
        'labels' in labelSchemaRevision && Array.isArray(labelSchemaRevision.labels)
            ? labelSchemaRevision.labels.length
            : undefined;

    const formattedStartedAt = job.started_at ? formatDateTime(job.started_at) : 'Waiting to start...';

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

                    <Text UNSAFE_className={classes.metaText}>{`Started: ${formattedStartedAt}`}</Text>
                    {device && <Text UNSAFE_className={classes.metaText}>{`Device: ${device}`}</Text>}
                </Flex>

                <Flex alignItems={'start'} direction={'column'} gap={'size-100'}>
                    {groupBy === 'architecture' ? (
                        <DatasetColumn datasetRevision={datasetRevision} labelsCount={labelsCount} />
                    ) : (
                        <ArchitectureColumn architectureId={modelArchitectureId} architecture={modelArchitecture} />
                    )}
                </Flex>

                <Flex gap={'size-100'} direction={'column'} alignItems={'center'}>
                    <ViewLogsButton jobId={job.job_id} />
                    {actions}
                </Flex>
            </Grid>
        </BottomProgressBar>
    );
};
