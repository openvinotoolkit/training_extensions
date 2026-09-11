// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Flex, Grid, repeat, Text, View } from '@geti-ui/ui';

import { distributeByLargestRemainder } from '../../../../utils';
import { LABEL_COLOR_MAPPING, SubsetTile } from './subset-distribution-stats.component';

import classes from './training-subsets.module.scss';

type SubsetDistributionRowProps = {
    existingSize: number;
    newSize: number;
    ariaLabel: string;
    percentage: number;
};

const SubsetLabel = ({ label, color }: { color: string; label: string }) => {
    return (
        <Flex alignItems={'center'} gap={'size-50'}>
            <SubsetTile color={color} />
            <Text>{label}:</Text>
        </Flex>
    );
};

const SubsetDistributionRow = ({ existingSize, newSize, ariaLabel, percentage }: SubsetDistributionRowProps) => {
    const resultingSize = existingSize + newSize;

    return (
        <>
            <Text>{existingSize}</Text>
            <Text>+</Text>
            <Text>{newSize}</Text>
            <Text>=</Text>
            <span aria-label={`${ariaLabel} result size`}>{resultingSize}</span>
            <span aria-label={`${ariaLabel} result percentage`}>({percentage}%)</span>
        </>
    );
};

type ResultingDatasetDistributionSubsetProps = {
    color: string;
    label: string;
    ariaLabel: string;
    newSize: number;
    existingSize: number;
    percentage: number;
};

const ResultingDatasetDistributionSubset = ({
    color,
    label,
    ariaLabel,
    existingSize,
    newSize,
    percentage,
}: ResultingDatasetDistributionSubsetProps) => {
    return (
        <>
            <SubsetLabel label={label} color={color} />

            <SubsetDistributionRow
                ariaLabel={ariaLabel}
                newSize={newSize}
                existingSize={existingSize}
                percentage={percentage}
            />
        </>
    );
};

type ResultingDatasetDistributionProps = {
    trainingSubsetSize: number;
    validationSubsetSize: number;
    testingSubsetSize: number;
    newTrainingSubsetSize: number;
    newValidationSubsetSize: number;
    newTestingSubsetSize: number;
};

const MAX_VALUE = 100;

export const ResultingDatasetDistribution = ({
    trainingSubsetSize,
    validationSubsetSize,
    testingSubsetSize,
    newTrainingSubsetSize,
    newValidationSubsetSize,
    newTestingSubsetSize,
}: ResultingDatasetDistributionProps) => {
    const { t } = useTranslation();
    const [trainingPercentage, validationPercentage, testingPercentage] = distributeByLargestRemainder(
        [
            trainingSubsetSize + newTrainingSubsetSize,
            validationSubsetSize + newValidationSubsetSize,
            testingSubsetSize + newTestingSubsetSize,
        ],
        MAX_VALUE
    );

    return (
        <Flex direction={'column'} gap={'size-50'}>
            <Text>{t('models.training.dataManagement.trainingSubsets.resultingDistribution')}</Text>
            <View backgroundColor={'static-gray-800'} borderRadius={'small'} padding={'size-100'}>
                <Grid
                    columns={[repeat(7, 'max-content')]}
                    alignItems={'center'}
                    columnGap={'size-100'}
                    rowGap={'size-50'}
                    UNSAFE_className={classes.resultingDistributionText}
                >
                    <ResultingDatasetDistributionSubset
                        label={t('dataset.revisions.trainingSubsets.training')}
                        ariaLabel={'Training'}
                        color={LABEL_COLOR_MAPPING.training}
                        newSize={newTrainingSubsetSize}
                        existingSize={trainingSubsetSize}
                        percentage={trainingPercentage}
                    />

                    <ResultingDatasetDistributionSubset
                        label={t('dataset.revisions.trainingSubsets.validation')}
                        ariaLabel={'Validation'}
                        color={LABEL_COLOR_MAPPING.validation}
                        newSize={newValidationSubsetSize}
                        existingSize={validationSubsetSize}
                        percentage={validationPercentage}
                    />

                    <ResultingDatasetDistributionSubset
                        label={t('dataset.revisions.trainingSubsets.test')}
                        ariaLabel={'Test'}
                        color={LABEL_COLOR_MAPPING.test}
                        newSize={newTestingSubsetSize}
                        existingSize={testingSubsetSize}
                        percentage={testingPercentage}
                    />
                </Grid>
            </View>
        </Flex>
    );
};
