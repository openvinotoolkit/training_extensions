// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { DatasetRevision } from '@/api/types';
import { useTranslation } from '@/i18n';
import { Flex, Text } from '@geti-ui/ui';
import { Image, Tag } from '@geti-ui/ui/icons';
import { useNumberFormatter } from 'react-aria';

import { formatDateTime } from '../../../../../shared/date-utils';
import { ModelBadge } from './model-badge.component';

import styles from './model-row.module.scss';

type DatasetColumnProps = {
    datasetRevision: DatasetRevision | undefined;
    labelsCount: number | undefined;
};

export const DatasetColumn = ({ datasetRevision, labelsCount }: DatasetColumnProps) => {
    const { t } = useTranslation();
    const totalCount = datasetRevision?.item_counts?.total;
    const formatter = useNumberFormatter();

    // Should never happen, but just in case
    if (datasetRevision === undefined) {
        return (
            <Flex alignItems={'center'} justifyContent={'center'}>
                {t('dataset.revisions.unknown')}
            </Flex>
        );
    }

    return (
        <Flex direction={'column'} gap={'size-50'}>
            <Text UNSAFE_className={styles.datasetRevisionName}>{datasetRevision.name}</Text>
            <Text UNSAFE_className={styles.datasetRevisionDate}>{formatDateTime(datasetRevision.created_at)}</Text>
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
