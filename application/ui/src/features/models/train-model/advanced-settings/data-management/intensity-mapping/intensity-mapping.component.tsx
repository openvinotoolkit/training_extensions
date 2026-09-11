// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction, useMemo } from 'react';

import type { ConfigurableParameter, ConfigurableParameterGroup, TrainingConfiguration } from '@/api/types';
import { useTranslation } from '@/i18n';

import { Accordion } from '../../components/accordion/accordion.component';
import { Parameters } from '../../components/parameters.component';
import { deepReplaceParameters, filterDependentParameters } from '../../utils';

type IntensityMappingProps = {
    intensityMappingParameters: ConfigurableParameterGroup;
    onTrainingConfigurationChange: Dispatch<SetStateAction<TrainingConfiguration | undefined>>;
};

const changeIntensityMappingParameter = (
    trainingConfiguration: TrainingConfiguration,
    newParameter: ConfigurableParameter
): TrainingConfiguration => {
    const parameters: TrainingConfiguration['parameters'] = deepReplaceParameters(
        trainingConfiguration.parameters,
        [newParameter],
        ['dataset_preparation', 'intensity_mapping']
    );

    return { parameters };
};

export const IntensityMapping = ({
    intensityMappingParameters,
    onTrainingConfigurationChange,
}: IntensityMappingProps) => {
    const { t } = useTranslation();
    const handleParameterChange = (parameter: ConfigurableParameter) => {
        onTrainingConfigurationChange((config) => {
            if (config === undefined) return;

            return changeIntensityMappingParameter(config, parameter);
        });
    };

    const parameters = useMemo(() => {
        return filterDependentParameters(intensityMappingParameters.parameters);
    }, [intensityMappingParameters.parameters]);

    return (
        <Accordion>
            <Accordion.Title>{t('models.training.dataManagement.intensityMapping.title')}</Accordion.Title>
            <Accordion.Content>
                <Accordion.Description>{intensityMappingParameters.description}</Accordion.Description>
                <Accordion.Divider marginY={'size-250'} />
                <Parameters.Container>
                    <Parameters parameters={parameters} onChange={handleParameterChange} />
                </Parameters.Container>
            </Accordion.Content>
        </Accordion>
    );
};
