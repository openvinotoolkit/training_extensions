// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Key, ReactNode } from 'react';

import type {
    ConfigurableParameter,
    ConfigurableParameterGroup,
    NumberEnumConfigurableParameter,
    StringEnumConfigurableParameter,
    TrainingConfigurationParameter,
} from '@/api/types';
import { useTranslation } from '@/i18n';
import {
    Content,
    ContextualHelp,
    DimensionValue,
    Flex,
    Grid,
    Item,
    minmax,
    Picker,
    Text,
    ToggleButtons,
    View,
} from '@geti-ui/ui';
import { isBoolean, isFunction } from 'lodash-es';

import { isParameter } from '../../../model-listing/model-training-parameters/utils';
import {
    isBoolEnableParameterGroup,
    isEnumNumberParameter,
    isEnumStringParameter,
    isNumberParameter,
    ParametersEnableGroupParameters,
} from '../utils';
import { BooleanParameterField } from './boolean-parameter-field.component';
import { NumberParameterField } from './number-parameter-field/number-parameter-field.component';
import { RangeParameterField } from './range-parameter-field/range-parameter-field.component';
import { ResetButton } from './reset-button.component';

import classes from './parameters.module.scss';

type ParametersProps = {
    parameters: TrainingConfigurationParameter[];
    onChange: (parameter: ConfigurableParameter, groupKeys?: string[]) => void;
    isReadOnly?: boolean;
    isDisabled?: boolean;
    marginStart?: DimensionValue;
    parentGroupKeys?: string[];
};

type ParametersGroupProps = {
    parametersGroup: ConfigurableParameterGroup;
    onChange: (parameter: ConfigurableParameter, groupKeys?: string[]) => void;
    isReadOnly?: boolean;
    isDisabled?: boolean;
    marginStart?: DimensionValue;
    parentGroupKeys?: string[];
};

type ParametersEnableGroupProps = {
    parameters: ParametersEnableGroupParameters;
    onChange: (parameter: ConfigurableParameter, groupKeys?: string[]) => void;
    isReadOnly: boolean;
    isDisabled?: boolean;
    parentGroupKeys?: string[];
};

type ParameterProps = {
    header: string;
    description: string;
    parameter: ConfigurableParameter;
    onChange: (parameter: ConfigurableParameter) => void;
    isDisabled?: boolean;
    marginStart?: DimensionValue;
    isReadOnly: boolean;
    isGroupHeader?: boolean;
};

type SingleParameterProps = {
    id?: string;
    header: string;
    description: string;
    parameter: ConfigurableParameter;
    onChange: (parameter: ConfigurableParameter) => void;
    isDisabled?: boolean;
    marginStart?: DimensionValue;
    isReadOnly: boolean;
};

type ParameterFieldProps = {
    parameter: ConfigurableParameter;
    onChange: (parameter: ConfigurableParameter) => void;
    isDisabled?: boolean;
};

type ParameterLayoutProps = {
    header: string;
    description: string;
    onReset?: () => void;
    children: ReactNode;
    marginStart?: DimensionValue;
    isGroupHeader?: boolean;
    ariaLabel?: string;
};

type ParameterNameProps = {
    name: string;
    description: string;
    gridColumn?: string;
    marginStart?: DimensionValue;
    isGroupHeader?: boolean;
};

const ParameterContextualHelp = ({ text }: { text: string }) => {
    return (
        <ContextualHelp variant='info'>
            <Content>
                <Text>{text}</Text>
            </Content>
        </ContextualHelp>
    );
};

const ParameterName = ({ name, description, marginStart, gridColumn, isGroupHeader = false }: ParameterNameProps) => {
    const parameterNameClass = isGroupHeader ? classes.parameterGroupHeader : classes.parameterName;

    return (
        <Text marginStart={marginStart} gridColumn={gridColumn} UNSAFE_className={parameterNameClass}>
            {name}
            <ParameterContextualHelp text={description} />
        </Text>
    );
};

const ParameterLayout = ({
    header,
    children,
    description,
    onReset,
    marginStart,
    isGroupHeader,
    ariaLabel,
}: ParameterLayoutProps) => {
    const resetAriaLabel = ariaLabel ?? header;

    return (
        <>
            <ParameterName
                name={header}
                description={description}
                gridColumn={'1/2'}
                marginStart={marginStart}
                isGroupHeader={isGroupHeader}
            />
            <View gridColumn={'2/3'}>{children}</View>
            {isFunction(onReset) && <ResetButton onPress={onReset} aria-label={`Reset ${resetAriaLabel}`} />}
        </>
    );
};

type ParameterReadOnlyProps = {
    parameter: Pick<ConfigurableParameter, 'value' | 'name' | 'description'>;
    marginStart?: DimensionValue;
};

type ParameterReadOnlyValueProps = Pick<ConfigurableParameter, 'value' | 'name'>;

const ParameterReadOnlyValue = ({ value, name }: ParameterReadOnlyValueProps) => {
    const { t } = useTranslation();

    if (isBoolean(value)) {
        return (
            <span aria-label={name}>
                {value ? t('models.training.parameters.on') : t('models.training.parameters.off')}
            </span>
        );
    }

    if (Array.isArray(value) && value.length === 2) {
        return (
            <span aria-label={name}>
                {value[0]} - {value[1]}
            </span>
        );
    }

    return <span aria-label={name}>{value}</span>;
};

const ParameterReadOnly = ({ parameter, marginStart }: ParameterReadOnlyProps) => {
    return (
        <ParameterLayout header={parameter.name} description={parameter.description} marginStart={marginStart}>
            <ParameterReadOnlyValue value={parameter.value} name={parameter.name} />
        </ParameterLayout>
    );
};

const StringEnumParameterField = ({
    parameter,
    onChange,
    isDisabled,
}: {
    parameter: StringEnumConfigurableParameter;
    onChange: (parameter: StringEnumConfigurableParameter) => void;
    isDisabled?: boolean;
}) => {
    const handleChange = (value: Key) => {
        onChange({
            ...parameter,
            value: value.toString(),
        });
    };

    const items = parameter.allowed_values.map((value) => ({ value }));

    return (
        <Picker
            isDisabled={isDisabled}
            items={items}
            selectedKey={parameter.value.toString()}
            onSelectionChange={(key) => key !== null && handleChange(key)}
            aria-label={`Select ${parameter.name}`}
        >
            {(item) => (
                <Item key={item.value} textValue={item.value}>
                    <Text>{item.value}</Text>
                </Item>
            )}
        </Picker>
    );
};

export const NumberEnumParameterField = ({
    parameter,
    onChange,
    isDisabled,
}: {
    parameter: NumberEnumConfigurableParameter;
    onChange: (parameter: NumberEnumConfigurableParameter) => void;
    isDisabled?: boolean;
}) => {
    const handleChange = (value: Key) => {
        onChange({
            ...parameter,
            value: Number(value),
        });
    };

    if (parameter.allowed_values.length < 4) {
        return (
            <ToggleButtons
                options={parameter.allowed_values}
                selectedOption={parameter.value}
                onOptionChange={handleChange}
                isDisabled={isDisabled}
            />
        );
    }

    const items = parameter.allowed_values.map((value) => ({ value }));

    return (
        <Picker
            items={items}
            selectedKey={parameter.value.toString()}
            onSelectionChange={(key) => key !== null && handleChange(key)}
            aria-label={`Select ${parameter.name}`}
        >
            {(item) => (
                <Item key={item.value} textValue={item.value.toString()}>
                    <Text>{item.value}</Text>
                </Item>
            )}
        </Picker>
    );
};

const ParameterField = ({ parameter, onChange, isDisabled }: ParameterFieldProps) => {
    if (isEnumStringParameter(parameter)) {
        return <StringEnumParameterField parameter={parameter} onChange={onChange} isDisabled={isDisabled} />;
    }

    if (isEnumNumberParameter(parameter)) {
        return <NumberEnumParameterField parameter={parameter} onChange={onChange} isDisabled={isDisabled} />;
    }

    if (isNumberParameter(parameter)) {
        const handleChange = (value: number) => {
            onChange({
                ...parameter,
                value,
            });
        };

        return (
            <NumberParameterField
                value={parameter.value}
                minValue={parameter.min_value ?? null}
                maxValue={parameter.max_value ?? null}
                onChange={handleChange}
                type={parameter.value_type}
                isDisabled={isDisabled}
                name={parameter.name}
            />
        );
    }

    if (parameter.value_type === 'float_range') {
        const handleChange = (value: [number, number]) => {
            onChange({
                ...parameter,
                value,
            });
        };

        return (
            <RangeParameterField
                value={parameter.value}
                maxValue={parameter.max_value}
                minValue={parameter.min_value}
                onChange={handleChange}
                isDisabled={isDisabled}
                name={parameter.name}
            />
        );
    }

    if (parameter.value_type === 'bool') {
        const handleChange = (value: boolean) => {
            onChange({
                ...parameter,
                value,
            });
        };

        return (
            <BooleanParameterField
                value={parameter.value}
                header={parameter.name}
                onChange={handleChange}
                isDisabled={isDisabled}
            />
        );
    }

    return null;
};

export const Parameter = ({
    header,
    description,
    parameter,
    onChange,
    isDisabled,
    marginStart,
    isReadOnly,
    isGroupHeader = false,
}: ParameterProps) => {
    if (isReadOnly) {
        return <ParameterReadOnly parameter={parameter} marginStart={marginStart} />;
    }

    const handleReset = () => {
        onChange({ ...parameter, value: parameter.default_value } as ConfigurableParameter);
    };

    return (
        <ParameterLayout
            header={header}
            description={description}
            onReset={handleReset}
            marginStart={marginStart}
            isGroupHeader={isGroupHeader}
        >
            <ParameterField parameter={parameter} onChange={onChange} isDisabled={isDisabled} />
        </ParameterLayout>
    );
};

const ParametersContainer = ({
    children,
    isReadOnly,
    columnGap = 'size-150',
    rowGap = 'size-350',
    id,
}: {
    children: ReactNode;
    isReadOnly?: boolean;
    columnGap?: DimensionValue;
    rowGap?: DimensionValue;
    id?: string;
}) => {
    const columns = isReadOnly ? ['size-3000', '1fr'] : ['size-3000', minmax(0, '1fr'), 'size-400'];

    return (
        <Grid
            columns={columns}
            columnGap={columnGap}
            rowGap={rowGap}
            alignItems={'center'}
            gridColumn={'1/-1'}
            data-testid={id}
        >
            {children}
        </Grid>
    );
};

const ParametersEnableGroup = ({
    parameters,
    onChange,
    isReadOnly,
    isDisabled,
    parentGroupKeys = [],
}: ParametersEnableGroupProps) => {
    const currentGroupKeys = [...parentGroupKeys, parameters.key];

    const handleChange = (parameter: ConfigurableParameter) => {
        onChange(parameter, currentGroupKeys);
    };

    const [enableParameter, ...configurableParameters] = parameters.parameters;

    return (
        <View UNSAFE_className={classes.parameterGroupContainer}>
            <ParametersContainer
                key={parameters.key}
                rowGap={configurableParameters.length > 0 ? 'size-100' : 'size-0'}
                isReadOnly={isReadOnly}
                id={createTestId(parentGroupKeys, parameters.key)}
            >
                <Parameter
                    header={parameters.name}
                    description={parameters.description}
                    parameter={enableParameter}
                    onChange={handleChange}
                    isReadOnly={isReadOnly}
                    isDisabled={isDisabled}
                    isGroupHeader
                />

                <Parameters
                    parameters={configurableParameters}
                    onChange={onChange}
                    isReadOnly={isReadOnly}
                    isDisabled={!enableParameter.value}
                    marginStart={'size-200'}
                    parentGroupKeys={currentGroupKeys}
                />
            </ParametersContainer>
        </View>
    );
};

const createTestId = (keys: string[] | undefined, parameterKey: string) => {
    if (keys === undefined) {
        return undefined;
    }

    if (keys.length > 0) {
        return `${keys.join('-')}-${parameterKey}`;
    }

    return parameterKey;
};

const ParametersGroup = ({
    parametersGroup,
    onChange,
    isReadOnly = false,
    isDisabled,
    marginStart,
    parentGroupKeys = [],
}: ParametersGroupProps) => {
    const currentGroupKeys = [...parentGroupKeys, parametersGroup.key];

    const handleChange = (parameter: ConfigurableParameter) => {
        onChange(parameter, currentGroupKeys);
    };

    if (isBoolEnableParameterGroup(parametersGroup)) {
        return (
            <ParametersEnableGroup
                parameters={parametersGroup}
                onChange={onChange}
                isReadOnly={isReadOnly}
                isDisabled={isDisabled}
                parentGroupKeys={parentGroupKeys}
            />
        );
    }

    return (
        <View UNSAFE_className={classes.parameterGroupContainer}>
            <Text UNSAFE_className={classes.parameterGroupTitle}>{parametersGroup.name}</Text>
            <Flex direction={'column'} gap={'size-300'}>
                {parametersGroup.parameters.map((parameter) => {
                    if (isParameter(parameter)) {
                        return (
                            <SingleParameter
                                id={createTestId(currentGroupKeys, parameter.key)}
                                key={parameter.key}
                                header={parameter.name}
                                description={parameter.description}
                                parameter={parameter}
                                onChange={handleChange}
                                isReadOnly={isReadOnly}
                                isDisabled={isDisabled}
                                marginStart={marginStart}
                            />
                        );
                    }

                    if (isBoolEnableParameterGroup(parameter)) {
                        return (
                            <ParametersEnableGroup
                                key={parameter.key}
                                parameters={parameter}
                                onChange={onChange}
                                isReadOnly={isReadOnly}
                                isDisabled={isDisabled}
                                parentGroupKeys={currentGroupKeys}
                            />
                        );
                    }

                    return (
                        <ParametersGroup
                            key={parameter.key}
                            parametersGroup={parameter}
                            onChange={onChange}
                            isReadOnly={isReadOnly}
                            isDisabled={isDisabled}
                            marginStart={marginStart}
                            parentGroupKeys={currentGroupKeys}
                        />
                    );
                })}
            </Flex>
        </View>
    );
};

const SingleParameter = ({
    id,
    parameter,
    onChange,
    isReadOnly,
    isDisabled,
    marginStart,
    header,
    description,
}: SingleParameterProps) => {
    return (
        <ParametersContainer key={parameter.name} id={id} isReadOnly={isReadOnly}>
            <Parameter
                header={header}
                description={description}
                parameter={parameter}
                onChange={onChange}
                isReadOnly={isReadOnly}
                isDisabled={isDisabled}
                marginStart={marginStart}
            />
        </ParametersContainer>
    );
};

export const Parameters = ({
    parameters,
    onChange,
    isReadOnly = false,
    isDisabled = false,
    marginStart,
    parentGroupKeys,
}: ParametersProps) => {
    return (
        <>
            {parameters.map((parameter) => {
                if (isParameter(parameter)) {
                    const handleChange = (changedParameter: ConfigurableParameter) => {
                        onChange(changedParameter, parentGroupKeys);
                    };

                    return (
                        <SingleParameter
                            id={createTestId(parentGroupKeys, parameter.key)}
                            key={parameter.key}
                            header={parameter.name}
                            description={parameter.description}
                            parameter={parameter}
                            onChange={handleChange}
                            isReadOnly={isReadOnly}
                            isDisabled={isDisabled}
                            marginStart={marginStart}
                        />
                    );
                }

                return (
                    <ParametersGroup
                        key={parameter.key}
                        parametersGroup={parameter}
                        onChange={onChange}
                        isReadOnly={isReadOnly}
                        isDisabled={isDisabled}
                        marginStart={marginStart}
                        parentGroupKeys={parentGroupKeys}
                    />
                );
            })}
        </>
    );
};

Parameters.Container = ParametersContainer;
Parameter.Layout = ParameterLayout;
