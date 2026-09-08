// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { DatasetRevision, Model, ModelArchitectureWithPerformanceCategory } from '@/api/types';
import { Badge, Flex, Grid, Text } from '@geti-ui/ui';

import { formatTrainingDateTime } from '../../../../../shared/date-utils';
import { formatBytes } from '../../../../../shared/util';
import { GRID_COLUMNS } from '../../constants';
import { AccuracyIndicator } from '../../model-variants/accuracy-indicator/accuracy-indicator.component';
import { type GroupByMode } from '../../types';
import { hasDeletedWeights, isFailedModel } from '../../utils/utils';
import { ParentRevisionModel } from '../parent-revision-model.component';
import { ArchitectureColumn } from './architecture-column.component';
import { DatasetColumn } from './dataset-revision-column.component';
import { getTestingMetric } from './utils';

import classes from './model-row.module.scss';

// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

type ModelRowProps = {
    model: Model;
    parentRevisionModel?: Model;
    onExpandModel?: (modelId: string) => void;
    groupBy: GroupByMode;
    datasetRevision: DatasetRevision | undefined;
    modelArchitecture: ModelArchitectureWithPerformanceCategory | undefined;
};

const FailedModel = () => {
    return <Badge variant={'negative'}>Failed</Badge>;
};

const DeletedWeightsModel = () => {
    return (
        <Badge
            variant={'yellow'}
            UNSAFE_style={{
                '--spectrum-yellow-background-color-default': `var(--brand-daisy)`,
            }}
        >
            Deleted weights
        </Badge>
    );
};

export const ModelRow = ({
    model,
    parentRevisionModel,
    onExpandModel,
    groupBy,
    datasetRevision,
    modelArchitecture,
}: ModelRowProps) => {
    const trainingEndTime = model.training_info.end_time;
    const totalSize = model.size;
    const device = model.training_info.device;
    const labelSchemaRevision = model.training_info.label_schema_revision ?? {};
    const labelsCount =
        'labels' in labelSchemaRevision && Array.isArray(labelSchemaRevision.labels)
            ? labelSchemaRevision.labels.length
            : undefined;

    const metricValue = getTestingMetric(model)?.value;

    return (
        <Grid columns={GRID_COLUMNS} alignItems={'center'} width={'100%'} columnGap={'size-200'}>
            <Flex direction={'column'} gap={'size-50'}>
                <Flex alignItems={'center'} gap={'size-100'} wrap>
                    <Text UNSAFE_className={classes.modelName} data-testid={'model-name'}>
                        {model.name}
                    </Text>
                    {isFailedModel(model) && <FailedModel />}
                    {hasDeletedWeights(model) && <DeletedWeightsModel />}
                </Flex>
                <Text UNSAFE_className={classes.secondaryText}>
                    {parentRevisionModel ? (
                        <ParentRevisionModel
                            id={parentRevisionModel.id}
                            name={parentRevisionModel.name}
                            onExpandModel={onExpandModel}
                        />
                    ) : (
                        <div />
                    )}
                </Text>
            </Flex>

            <Text UNSAFE_className={classes.dateText}>{formatTrainingDateTime(trainingEndTime)}</Text>

            {groupBy === 'architecture' ? (
                <DatasetColumn datasetRevision={datasetRevision} labelsCount={labelsCount} />
            ) : (
                <ArchitectureColumn architectureId={model.architecture} architecture={modelArchitecture} />
            )}

            <Text UNSAFE_className={classes.smallText} data-testid={'device info'}>
                {device ? device.name : '-'}
            </Text>

            <Text UNSAFE_className={classes.smallText} data-testid={'model size'}>
                {totalSize > 0 ? formatBytes(totalSize) : '-'}
            </Text>

            {metricValue === undefined ? <Text>-</Text> : <AccuracyIndicator accuracy={metricValue} />}
        </Grid>
    );
};
