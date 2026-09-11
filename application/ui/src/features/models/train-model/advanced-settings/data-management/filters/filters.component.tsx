// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction } from 'react';

import type { ConfigurableParameterGroup, TrainingConfiguration } from '@/api/types';
import { useTranslation } from '@/i18n';
import { Grid, minmax } from '@geti-ui/ui';

import { Accordion } from '../../components/accordion/accordion.component';
import { deepReplaceParameters } from '../../utils';
import { FiltersOptions } from './filters-options.component';
import {
    checkIfFiltersAreEnabled,
    isFilterConfigurableParameterGroup,
    type FilterConfigurableParameters,
} from './utils';

type FiltersProps = {
    filtersParameters: ConfigurableParameterGroup;
    onTrainingConfigurationChange: Dispatch<SetStateAction<TrainingConfiguration | undefined>>;
};

const changeFilterParameters = (
    trainingConfiguration: TrainingConfiguration,
    { key, newParameters }: { key: string; newParameters: FilterConfigurableParameters }
): TrainingConfiguration => ({
    parameters: deepReplaceParameters(trainingConfiguration.parameters, newParameters, [
        'dataset_preparation',
        'filtering',
        key,
    ]),
});

export const Filters = ({ filtersParameters, onTrainingConfigurationChange }: FiltersProps) => {
    const { t } = useTranslation();

    const handleFilterChange = (key: string, newParameters: FilterConfigurableParameters) => {
        onTrainingConfigurationChange((config) => {
            if (config === undefined) return;

            return changeFilterParameters(config, { key, newParameters });
        });
    };

    const parameters = filtersParameters.parameters.filter(isFilterConfigurableParameterGroup);

    const areFiltersEnabled = checkIfFiltersAreEnabled(parameters);

    return (
        <Accordion>
            <Accordion.Title>
                {t('models.training.dataManagement.filters.title')}{' '}
                <Accordion.Tag ariaLabel={'Filters tag'}>
                    {areFiltersEnabled ? t('models.training.parameters.on') : t('models.training.parameters.off')}
                </Accordion.Tag>
            </Accordion.Title>
            <Accordion.Content>
                <Accordion.Description>{filtersParameters.description}</Accordion.Description>
                <Accordion.Divider marginY={'size-250'} />
                <Grid
                    columns={['size-3000', minmax('size-3400', '1fr'), 'size-400']}
                    gap={'size-300'}
                    alignItems={'center'}
                >
                    <FiltersOptions filterParameters={parameters} onFilterChange={handleFilterChange} />
                </Grid>
            </Accordion.Content>
        </Accordion>
    );
};
