// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Fragment } from 'react';

import type { TrainingConfigurationParameter } from '@/api/types';
import { useTranslation } from '@/i18n';
import { Grid, Text } from '@geti-ui/ui';

import { useGetModelTrainingConfiguration } from '../../hooks/api/use-get-model-training-configuration.hook';
import { filterDependentParameters } from '../../train-model/advanced-settings/utils';
import { Box } from '../components/box/box.component';
import { findGroupByKey, flattenParameters } from './utils';

import classes from './model-training.module.scss';

type ModelTrainingParametersProps = {
    modelId: string;
};

type TrainingConfigurationParametersListProps = {
    parameters: TrainingConfigurationParameter[];
};

const TrainingConfigurationParametersList = ({ parameters }: TrainingConfigurationParametersListProps) => {
    const { t } = useTranslation();
    const parameterRows = flattenParameters(parameters, t);

    if (parameterRows.length === 0) {
        return <Text>{t('models.training.parameters.noParameters')}</Text>;
    }

    return (
        <Grid columns={['1fr', '1fr']} gap={'size-100'}>
            {parameterRows.map((row, index) => (
                <Fragment key={`${index}-${row.isGroup}-${row.depth}-${row.name}-${row.value}`}>
                    <Text
                        UNSAFE_style={{
                            paddingInlineStart: `calc(${row.depth} * var(--spectrum-global-dimension-size-200))`,
                        }}
                    >
                        {row.isGroup ? row.name : `• ${row.name}`}
                    </Text>

                    <Text>{row.isGroup ? '' : row.value}</Text>
                </Fragment>
            ))}
        </Grid>
    );
};

export const ModelTrainingParameters = ({ modelId }: ModelTrainingParametersProps) => {
    const { t } = useTranslation();
    const { data } = useGetModelTrainingConfiguration(modelId);

    const trainingGroup = findGroupByKey(data?.parameters, 'training');
    const datasetPreparationGroup = findGroupByKey(data?.parameters, 'dataset_preparation');
    const filteringGroup = findGroupByKey(datasetPreparationGroup?.parameters, 'filtering');
    const augmentationGroup = findGroupByKey(datasetPreparationGroup?.parameters, 'augmentation');
    const intensityMappingGroup = findGroupByKey(datasetPreparationGroup?.parameters, 'intensity_mapping');

    const learningParameters = filterDependentParameters(trainingGroup?.parameters ?? []);
    const intensityMappingParameters = filterDependentParameters(intensityMappingGroup?.parameters ?? []);

    return (
        <Grid columns={['1fr', '1fr', '1fr']} gap={'size-200'}>
            <Box
                testId={'Box-LEARNING PARAMETERS'}
                contentClassName={classes.scrollableContent}
                title={t('models.training.parameters.learningParameters')}
                content={<TrainingConfigurationParametersList parameters={learningParameters} />}
            />
            <Box
                testId={'Box-FILTERS'}
                contentClassName={classes.scrollableContent}
                title={t('models.training.parameters.filters')}
                content={<TrainingConfigurationParametersList parameters={filteringGroup?.parameters || []} />}
            />
            <Box
                testId={'Box-AUGMENTATIONS'}
                contentClassName={classes.scrollableContent}
                title={t('models.training.parameters.augmentations')}
                content={<TrainingConfigurationParametersList parameters={augmentationGroup?.parameters || []} />}
            />
            <Box
                testId={'Box-INTENSITY MAPPING'}
                contentClassName={classes.scrollableContent}
                title={t('models.training.parameters.intensityMapping')}
                content={<TrainingConfigurationParametersList parameters={intensityMappingParameters} />}
            />
        </Grid>
    );
};
