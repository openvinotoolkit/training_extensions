// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction, useState } from 'react';

import type { ConfigurableParameter, TrainingConfiguration } from '@/api/types';
import { useTranslation } from '@/i18n';
import { Content, Flex, Heading, InlineAlert, View } from '@geti-ui/ui';
import { useGetDatasetItems } from 'hooks/use-get-dataset-items.hook';
import { isEqual } from 'lodash-es';

import { isParameterGroup } from '../../../../model-listing/model-training-parameters/utils';
import { distributeByLargestRemainder } from '../../../../utils';
import { Accordion } from '../../components/accordion/accordion.component';
import { ResultingDatasetDistribution } from './resulting-dataset-distribution.component';
import { SubsetsDistribution } from './subset-distribution.component';
import {
    getSubsets,
    MAX_RATIO_VALUE,
    SubsetSplitParameters,
    TEST_SUBSET_KEY,
    TRAINING_SUBSET_KEY,
    VALIDATION_SUBSET_KEY,
} from './utils';

type TrainingSubsetsProps = {
    defaultSubsetParameters: SubsetSplitParameters;
    subsetsParameters: SubsetSplitParameters;
    onTrainingConfigurationChange: Dispatch<SetStateAction<TrainingConfiguration | undefined>>;
};

const TrainingSubsetsUnavailable = () => {
    const { t } = useTranslation();

    return (
        <InlineAlert variant={'notice'} marginTop={'size-200'}>
            <Heading>{t('models.training.dataManagement.trainingSubsets.unavailableTitle')}</Heading>
            <Content>
                {t('models.training.dataManagement.trainingSubsets.unavailableDescription')}
                <br />
                {t('models.training.dataManagement.trainingSubsets.unavailableHint')}
            </Content>
        </InlineAlert>
    );
};

const updateSubsetSplitValues = (
    config: TrainingConfiguration,
    getValueForKey: (parameter: ConfigurableParameter) => number
): TrainingConfiguration => ({
    parameters: config.parameters.map((parameterGroup) => {
        if (parameterGroup.key !== 'dataset_preparation' || !isParameterGroup(parameterGroup)) {
            return parameterGroup;
        }
        return {
            ...parameterGroup,
            parameters: parameterGroup.parameters.map((parameter) => {
                if (parameter.key !== 'subset_split' || !isParameterGroup(parameter)) {
                    return parameter;
                }
                return {
                    ...parameter,
                    parameters: parameter.parameters.map((subsetSplitParameter) => {
                        if (
                            isParameterGroup(subsetSplitParameter) ||
                            ![TRAINING_SUBSET_KEY, VALIDATION_SUBSET_KEY, TEST_SUBSET_KEY].includes(
                                subsetSplitParameter.key
                            )
                        ) {
                            return subsetSplitParameter;
                        }
                        return {
                            ...subsetSplitParameter,
                            value: getValueForKey(subsetSplitParameter),
                        } as ConfigurableParameter;
                    }),
                };
            }),
        };
    }),
});

const useSubsetDatasetSizes = () => {
    const { totalCount: trainingSubsetSize } = useGetDatasetItems({
        annotationStatus: 'with_annotations',
        subsets: ['training'],
    });
    const { totalCount: testingSubsetSize } = useGetDatasetItems({
        annotationStatus: 'with_annotations',
        subsets: ['testing'],
    });
    const { totalCount: validationSubsetSize } = useGetDatasetItems({
        annotationStatus: 'with_annotations',
        subsets: ['validation'],
    });
    const { totalCount: unassignedSubsetSize } = useGetDatasetItems({
        annotationStatus: 'with_annotations',
        subsets: ['unassigned'],
    });

    const assignedDatasetItemsSize = trainingSubsetSize + testingSubsetSize + validationSubsetSize;
    const totalDatasetItemsSize = assignedDatasetItemsSize + unassignedSubsetSize;

    return {
        trainingSubsetSize,
        testingSubsetSize,
        validationSubsetSize,
        unassignedSubsetSize,
        assignedDatasetItemsSize,
        totalDatasetItemsSize,
    };
};

export const TrainingSubsets = ({
    defaultSubsetParameters,
    subsetsParameters,
    onTrainingConfigurationChange,
}: TrainingSubsetsProps) => {
    const { t } = useTranslation();
    const { trainingSubset, validationSubset } = getSubsets(subsetsParameters);
    const {
        validationSubsetSize,
        testingSubsetSize,
        unassignedSubsetSize,
        assignedDatasetItemsSize,
        totalDatasetItemsSize,
        trainingSubsetSize,
    } = useSubsetDatasetSizes();

    const areTrainingSubsetParametersChanged = !isEqual(defaultSubsetParameters, subsetsParameters);

    const [subsetsDistribution, setSubsetsDistribution] = useState<number[]>([
        trainingSubset.value,
        trainingSubset.value + validationSubset.value,
    ]);

    const trainingSubsetRatio = subsetsDistribution[0];
    const validationSubsetRatio = subsetsDistribution[1] - trainingSubsetRatio;
    const testSubsetRatio = MAX_RATIO_VALUE - subsetsDistribution[1];

    const handleUpdateSubsetsConfiguration = (values: number[]): void => {
        const trainingSubsetValue = values[0];
        const validationSubsetValue = values[1] - trainingSubsetValue;
        const testSubsetValue = MAX_RATIO_VALUE - values[1];

        const KEY_VALUE_MAP: Record<string, number> = {
            [TRAINING_SUBSET_KEY]: trainingSubsetValue,
            [VALIDATION_SUBSET_KEY]: validationSubsetValue,
            [TEST_SUBSET_KEY]: testSubsetValue,
        };

        onTrainingConfigurationChange((config) => {
            if (!config?.parameters) return undefined;
            return updateSubsetSplitValues(config, (parameter) => KEY_VALUE_MAP[parameter.key]);
        });
    };

    const handleSubsetsConfigurationReset = (): void => {
        setSubsetsDistribution([
            trainingSubset.default_value,
            trainingSubset.default_value + validationSubset.default_value,
        ]);

        onTrainingConfigurationChange((config) => {
            if (!config?.parameters) return undefined;
            return updateSubsetSplitValues(config, (parameter) => parameter.default_value as number);
        });
    };

    const [newTrainingSubsetSize, newValidationSubsetSize, newTestingSubsetSize] = distributeByLargestRemainder(
        [trainingSubsetRatio, validationSubsetRatio, testSubsetRatio],
        unassignedSubsetSize
    );

    const areSubsetsSizesValid = () => {
        const resultingTrainingSubsetSize = trainingSubsetSize + newTrainingSubsetSize;
        const resultingValidationSubsetSize = validationSubsetSize + newValidationSubsetSize;
        const resultingTestingSubsetSize = testingSubsetSize + newTestingSubsetSize;
        return ![resultingTrainingSubsetSize, resultingValidationSubsetSize, resultingTestingSubsetSize].some(
            (size) => size === 0
        );
    };

    const subsetsSizesInvalid = areTrainingSubsetParametersChanged && !areSubsetsSizesValid();

    return (
        <Accordion>
            <Accordion.Title>
                {t('models.training.dataManagement.trainingSubsets.title')}{' '}
                <Accordion.Tag ariaLabel={'Training subsets tag'}>
                    {trainingSubsetRatio}/{validationSubsetRatio}/{testSubsetRatio}%
                </Accordion.Tag>
            </Accordion.Title>
            <Accordion.Content>
                <Accordion.Description>
                    {t('models.training.dataManagement.trainingSubsets.description')}
                </Accordion.Description>
                <Accordion.Divider marginY={'size-200'} />
                <View>
                    <span aria-label={'Total dataset samples'}>
                        {t('models.training.dataManagement.trainingSubsets.datasetSamples', {
                            count: totalDatasetItemsSize,
                        })}
                    </span>
                    <Flex alignItems={'center'} gap={'size-100'}>
                        <span aria-label={'Total assigned samples'}>
                            {t('models.training.dataManagement.trainingSubsets.assignedSamples', {
                                count: assignedDatasetItemsSize,
                            })}
                        </span>
                        <span aria-label={'Total unassigned samples'}>
                            {t('models.training.dataManagement.trainingSubsets.unassignedSamples', {
                                count: unassignedSubsetSize,
                            })}
                        </span>
                    </Flex>
                    <Accordion.Divider marginY={'size-200'} />
                    <SubsetsDistribution
                        subsetsDistribution={subsetsDistribution}
                        onSubsetsDistributionChange={setSubsetsDistribution}
                        testSubsetSize={newTestingSubsetSize}
                        trainingSubsetSize={newTrainingSubsetSize}
                        validationSubsetSize={newValidationSubsetSize}
                        onSubsetsDistributionChangeEnd={handleUpdateSubsetsConfiguration}
                        onSubsetsDistributionReset={handleSubsetsConfigurationReset}
                    />
                    <Accordion.Divider marginY={'size-200'} />
                    <ResultingDatasetDistribution
                        trainingSubsetSize={trainingSubsetSize}
                        validationSubsetSize={validationSubsetSize}
                        testingSubsetSize={testingSubsetSize}
                        newTrainingSubsetSize={newTrainingSubsetSize}
                        newValidationSubsetSize={newValidationSubsetSize}
                        newTestingSubsetSize={newTestingSubsetSize}
                    />
                </View>

                {subsetsSizesInvalid && <TrainingSubsetsUnavailable />}
            </Accordion.Content>
        </Accordion>
    );
};
