// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { Checkbox, Content, ContextualHelp, Flex, NumberField, Text } from '@geti-ui/ui';

import { ResetButton } from '../../components/reset-button.component';
import { FilterConfigurableParameterGroup, FilterConfigurableParameters } from './utils';

type FilterOptionTooltipProps = {
    description: string;
};

const FilterOptionContextualHelp = ({ description }: FilterOptionTooltipProps) => {
    return (
        <ContextualHelp variant='info'>
            <Content>
                <Text>{description}</Text>
            </Content>
        </ContextualHelp>
    );
};

type FilterOptionProps = {
    filterParameter: FilterConfigurableParameterGroup;
    onFilterChange: (newFilterConfigurableParameters: FilterConfigurableParameters) => void;
};

const FilterOption = ({ filterParameter, onFilterChange }: FilterOptionProps) => {
    const { t } = useTranslation();
    const { description, name, parameters, key } = filterParameter;
    const [enableParameter, configurableParameter] = parameters;
    const isUnlimited = !enableParameter.value;

    const handleToggleChange = (newIsUnlimited: boolean) => {
        onFilterChange([{ ...enableParameter, value: !newIsUnlimited }, configurableParameter]);
    };

    const handleParameterChange = (newValue: number) => {
        onFilterChange([enableParameter, { ...configurableParameter, value: newValue }]);
    };

    const handleReset = () => {
        onFilterChange([
            { ...enableParameter, value: enableParameter.default_value },
            { ...configurableParameter, value: configurableParameter.default_value },
        ]);
    };

    const toggleName = key.toLocaleLowerCase().includes('min')
        ? t('models.training.dataManagement.filters.noMinimum')
        : key.toLocaleLowerCase().includes('max')
          ? t('models.training.dataManagement.filters.noMaximum')
          : t('models.training.dataManagement.filters.unlimited');

    return (
        <>
            <Text gridColumn={'1/2'}>
                {name} <FilterOptionContextualHelp description={description} />
            </Text>
            <Flex gap={'size-200'} gridColumn={'2/3'}>
                <NumberField
                    aria-label={`Change ${configurableParameter.name}`}
                    minValue={configurableParameter.min_value ?? undefined}
                    maxValue={configurableParameter.max_value ?? undefined}
                    step={1}
                    value={configurableParameter.value}
                    isDisabled={isUnlimited}
                    onChange={handleParameterChange}
                    hideStepper
                    width={'size-900'}
                />

                <Checkbox
                    isEmphasized
                    isSelected={isUnlimited}
                    onChange={handleToggleChange}
                    aria-label={`Toggle ${configurableParameter.name}`}
                >
                    {toggleName}
                </Checkbox>
            </Flex>

            <ResetButton gridColumn={'3/4'} onPress={handleReset} aria-label={`Reset ${configurableParameter.name}`} />
        </>
    );
};

type FiltersOptionsProps = {
    filterParameters: FilterConfigurableParameterGroup[];
    onFilterChange: (key: string, newFilterConfigurableParameters: FilterConfigurableParameters) => void;
};

export const FiltersOptions = ({ filterParameters, onFilterChange }: FiltersOptionsProps) => {
    return (
        <>
            {filterParameters.map((filterParameter) => (
                <FilterOption
                    key={filterParameter.key}
                    filterParameter={filterParameter}
                    onFilterChange={(parameters) => onFilterChange(filterParameter.key, parameters)}
                />
            ))}
        </>
    );
};
