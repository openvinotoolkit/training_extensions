// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Flex, Text } from '@geti-ui/ui';

import { distributeByLargestRemainder } from '../../../utils';

import classes from './three-section-range.module.scss';

type ThreeSectionRangeProps = {
    id?: string;
    trainingValue: number;
    validationValue: number;
    testingValue: number;
};

const MAX_VALUE = 100;

export const ThreeSectionRange = ({ id, trainingValue, validationValue, testingValue }: ThreeSectionRangeProps) => {
    const { t } = useTranslation();
    const [trainingPercentage, validationPercentage, testingPercentage] = distributeByLargestRemainder(
        [trainingValue, validationValue, testingValue],
        MAX_VALUE
    );

    const labelledPercentages = [
        {
            label: t('dataset.revisions.trainingSubsets.training'),
            percentage: trainingPercentage,
            color: 'var(--training-subset)',
        },
        {
            label: t('dataset.revisions.trainingSubsets.validation'),
            percentage: validationPercentage,
            color: 'var(--validation-subset)',
        },
        {
            label: t('dataset.revisions.trainingSubsets.test'),
            percentage: testingPercentage,
            color: 'var(--test-subset)',
        },
    ];

    return (
        <Flex alignItems={'center'} data-testid={id}>
            <Flex gap={'size-150'} alignItems={'center'} UNSAFE_className={classes.label}>
                {labelledPercentages.map(({ label, percentage, color }) => (
                    <Flex key={label} gap={'size-75'} alignItems={'center'}>
                        <span
                            style={{
                                width: 8,
                                height: 8,
                                borderRadius: '50%',
                                backgroundColor: color,
                                display: 'inline-block',
                                flexShrink: 0,
                            }}
                        />
                        <Text UNSAFE_className={classes.label}>{`${label} ${percentage}%`}</Text>
                    </Flex>
                ))}
            </Flex>
        </Flex>
    );
};
