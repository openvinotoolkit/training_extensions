// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Flex, Text, View } from '@geti-ui/ui';

import classes from './training-subsets.module.scss';

type SubsetDistributionStatsProps = {
    trainingSize: number;
    validationSize: number;
    testSize: number;
    totalSize: number;
};

export const LABEL_COLOR_MAPPING = {
    training: 'var(--training-subset)',
    validation: 'var(--validation-subset)',
    test: 'var(--test-subset)',
};

export const SubsetTile = ({ color }: { color: string }) => {
    return (
        <View height={'size-100'} width={'size-100'} borderRadius={'small'} UNSAFE_style={{ backgroundColor: color }} />
    );
};

const SubsetDistributionStat = ({
    size,
    color,
    title,
    ariaLabel,
}: {
    size: number;
    color: string;
    title: string;
    ariaLabel: string;
}) => {
    return (
        <Flex alignItems={'center'} gap={'size-50'}>
            <SubsetTile color={color} />
            <span aria-label={`${ariaLabel} subset size`}>
                {title}: {size}
            </span>
        </Flex>
    );
};

export const SubsetDistributionStats = ({
    trainingSize,
    validationSize,
    testSize,
    totalSize,
}: SubsetDistributionStatsProps) => {
    const { t } = useTranslation();

    return (
        <View gridArea={'counts'} backgroundColor={'static-gray-800'} borderRadius={'small'} padding={'size-100'}>
            <Flex alignItems={'center'} justifyContent={'space-between'} UNSAFE_className={classes.statsText}>
                <Flex alignItems={'center'} gap={'size-200'}>
                    <SubsetDistributionStat
                        title={t('dataset.revisions.trainingSubsets.training')}
                        ariaLabel={'Training'}
                        color={LABEL_COLOR_MAPPING.training}
                        size={trainingSize}
                    />
                    <SubsetDistributionStat
                        title={t('dataset.revisions.trainingSubsets.validation')}
                        ariaLabel={'Validation'}
                        color={LABEL_COLOR_MAPPING.validation}
                        size={validationSize}
                    />
                    <SubsetDistributionStat
                        title={t('dataset.revisions.trainingSubsets.test')}
                        ariaLabel={'Test'}
                        color={LABEL_COLOR_MAPPING.test}
                        size={testSize}
                    />
                </Flex>
                <Text>
                    <Text UNSAFE_className={classes.totalStats}>
                        {t('models.training.dataManagement.trainingSubsets.totalLabel')}{' '}
                    </Text>
                    <span aria-label={'Total size'}>{totalSize}</span>{' '}
                    {t('models.training.dataManagement.trainingSubsets.mediaItemsLabel')}
                </Text>
            </Flex>
        </View>
    );
};
