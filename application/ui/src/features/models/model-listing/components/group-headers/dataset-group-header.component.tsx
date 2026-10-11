// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { dimensionValue, Flex, Grid, Heading, Text } from '@geti-ui/ui';
import { Image, Tag } from '@geti-ui/ui/icons';
import { DATE_TIME_FORMAT, useFormatDate } from 'hooks/use-format-date.hook';
import { useNumberFormatter } from 'react-aria';

import type { DatasetGroup } from '../../types';
import { DatasetActions } from '../dataset-actions/dataset-actions.component';
import { ModelBadge } from '../model-row/model-badge.component';
import { ThreeSectionRange } from '../three-section-range/three-section-range.component';

type DatasetGroupHeaderProps = {
    dataset: DatasetGroup;
};

export const DatasetGroupHeader = ({ dataset }: DatasetGroupHeaderProps) => {
    const hasDatasetRevisionData = dataset.imageCount > 0 && !dataset.filesDeleted;
    const gridColumns = hasDatasetRevisionData ? ['auto', '1fr', 'auto', 'max-content'] : ['auto', '1fr', 'auto'];
    const formatter = useNumberFormatter();
    const { t } = useTranslation();
    const formatDate = useFormatDate(DATE_TIME_FORMAT);
    const createdAt = formatDate(dataset.createdAt);

    return (
        <Grid columns={gridColumns} alignItems={'center'} marginBottom={'size-225'} gap={'size-200'}>
            <Flex alignItems={'center'} gap={'size-50'}>
                <Heading level={2} UNSAFE_style={{ fontSize: dimensionValue('size-300') }}>
                    {dataset.name}
                </Heading>

                <DatasetActions dataset={dataset} />
            </Flex>
            <Text
                UNSAFE_style={{
                    color: 'var(--spectrum-global-color-gray-700)',
                }}
            >
                {createdAt === null ? '-' : t('dataset.revisions.createdOn', { date: createdAt })}
            </Text>

            <Flex gap={'size-50'} justifyContent={'center'}>
                <ModelBadge>
                    <Tag /> {dataset.labelCount}
                </ModelBadge>
                <ModelBadge>
                    <Image /> {formatter.format(dataset.imageCount)}
                </ModelBadge>
            </Flex>

            {hasDatasetRevisionData && (
                <ThreeSectionRange
                    id={`dataset-range-${dataset.id}`}
                    trainingValue={dataset.trainingSubsets.training}
                    validationValue={dataset.trainingSubsets.validation}
                    testingValue={dataset.trainingSubsets.testing}
                />
            )}
        </Grid>
    );
};
