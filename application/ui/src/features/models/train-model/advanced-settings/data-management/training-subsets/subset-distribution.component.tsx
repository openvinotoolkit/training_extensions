// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Grid, minmax, View } from '@geti-ui/ui';

import { ResetButton } from '../../components/reset-button.component';
import { SubsetDistributionStats } from './subset-distribution-stats.component';
import { SubsetsDistributionSlider } from './subsets-distribution-slider/subsets-distribution-slider.component';

type SubsetsDistributionProps = {
    trainingSubsetSize: number;
    validationSubsetSize: number;
    testSubsetSize: number;
    subsetsDistribution: number[];
    onSubsetsDistributionChange: (values: number[]) => void;
    onSubsetsDistributionChangeEnd: (values: number[]) => void;
    onSubsetsDistributionReset: () => void;
};

export const SubsetsDistribution = ({
    subsetsDistribution,
    trainingSubsetSize,
    testSubsetSize,
    validationSubsetSize,
    onSubsetsDistributionChange,
    onSubsetsDistributionChangeEnd,
    onSubsetsDistributionReset,
}: SubsetsDistributionProps) => {
    const { t } = useTranslation();

    const handleSubsetDistributionChange = (values: number[] | number): void => {
        if (Array.isArray(values)) {
            onSubsetsDistributionChange(values);
        }
    };

    const handleSubsetDistributionChangeEnd = (values: number[] | number): void => {
        if (Array.isArray(values)) {
            onSubsetsDistributionChangeEnd(values);
        }
    };

    return (
        <View>
            <Grid
                areas={['label label', 'slider reset', 'counts counts']}
                columns={[minmax('size-3400', '1fr'), 'max-content']}
                alignItems={'center'}
                columnGap={'size-250'}
            >
                <SubsetsDistributionSlider
                    aria-label={'Distribute samples'}
                    minValue={0}
                    maxValue={100}
                    step={1}
                    value={[subsetsDistribution[0], subsetsDistribution[1]]}
                    onChange={handleSubsetDistributionChange}
                    onChangeEnd={handleSubsetDistributionChangeEnd}
                    label={t('models.training.dataManagement.trainingSubsets.distributionLabel')}
                />
                <ResetButton
                    gridArea={'reset'}
                    onPress={onSubsetsDistributionReset}
                    aria-label={'Reset training subsets'}
                />
                <SubsetDistributionStats
                    testSize={testSubsetSize}
                    trainingSize={trainingSubsetSize}
                    validationSize={validationSubsetSize}
                    totalSize={trainingSubsetSize + validationSubsetSize + testSubsetSize}
                />
            </Grid>
        </View>
    );
};
