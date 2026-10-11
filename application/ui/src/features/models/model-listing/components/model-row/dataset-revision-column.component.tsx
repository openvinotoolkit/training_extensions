// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { DatasetRevision } from '@/api/types';
import { useTranslation } from '@/i18n';
import { Flex, Text } from '@geti-ui/ui';
import { Image, Tag } from '@geti-ui/ui/icons';
import { DATE_TIME_FORMAT, useFormatDate } from 'hooks/use-format-date.hook';
import { useNumberFormatter } from 'react-aria';

import { ModelBadge } from './model-badge.component';

import styles from './model-row.module.scss';

type DatasetColumnProps = {
    datasetRevision: DatasetRevision | undefined;
    labelsCount: number | undefined;
    /** Shown while no revision exists yet, e.g. a training job that was started on a dataset view. */
    pendingDatasetName?: string;
};

export const DatasetColumn = ({ datasetRevision, labelsCount, pendingDatasetName }: DatasetColumnProps) => {
    const { t } = useTranslation();
    const totalCount = datasetRevision?.item_counts?.total;
    const formatter = useNumberFormatter();
    const formatDate = useFormatDate(DATE_TIME_FORMAT);

    if (datasetRevision === undefined) {
        return <Flex alignItems={'center'}>{pendingDatasetName ?? t('common.labels.unknown')}</Flex>;
    }

    return (
        <Flex direction={'column'} gap={'size-50'}>
            <Text UNSAFE_className={styles.datasetRevisionName}>{datasetRevision.name}</Text>
            <Text UNSAFE_className={styles.datasetRevisionDate}>{formatDate(datasetRevision.created_at) ?? '-'}</Text>
            <Flex gap={'size-100'}>
                {labelsCount !== undefined && (
                    <ModelBadge id={'labels-count'}>
                        <Tag />
                        <Text>{labelsCount}</Text>
                    </ModelBadge>
                )}
                {totalCount !== undefined && (
                    <ModelBadge id={'dataset-count'}>
                        <Image />
                        <Text>{formatter.format(totalCount)}</Text>
                    </ModelBadge>
                )}
            </Flex>
        </Flex>
    );
};
