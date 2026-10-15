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

type SharedParametersProps = {
    onChange: (parameter: ConfigurableParameter, groupKeys?: string[]) => void;
    isReadOnly?: boolean;
    isDisabled?: boolean;
    marginStart?: DimensionValue;
    parentGroupKeys?: string[];
};

type ParameterNameProps = {
    name: string;
    description: string;
    gridColumn?: string;
    marginStart?: DimensionValue;
    isGroupHeader?: boolean;
};

const ParameterName = ({ name, description, marginStart, gridColumn, isGroupHeader = false }: ParameterNameProps) => (
    <Text
        marginStart={marginStart}
        gridColumn={gridColumn}
        UNSAFE_className={isGroupHeader ? classes.parameterGroupHeader : classes.parameterName}
    >
        {name}
        <ContextualHelp variant='info'>
            <Content>
                <Text>{description}</Text>
            </Content>
        </ContextualHelp>
    </Text>
);

type ParameterLayoutProps = {
    header: string;
    description: string;
    onReset?: () => void;
    children: ReactNode;
    marginStart?: DimensionValue;
    isGroupHeader?: boolean;
    ariaLabel?: string;
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

type ParametersContainerProps = {
    children: ReactNode;
    isReadOnly?: boolean;
    columnGap?: DimensionValue;
    rowGap?: DimensionValue;
    id?: string;
};

const ParametersContainer = ({
    children,
    isReadOnly,
    columnGap = 'size-150',
    rowGap = 'size-350',
    id,
}: ParametersContainerProps) => {
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

type ParameterReadOnlyProps = {
    parameter: Pick<ConfigurableParameter, 'value' | 'name' | 'description'>;
    marginStart?: DimensionValue;
};

type ParameterReadOnlyValueProps = Pick<ConfigurableParameter, 'value' | 'name'>;

const ParameterReadOnlyValue = ({ value, name }: ParameterReadOnlyValueProps) => {
    const { t } = useTranslation();

    if (isBoolean(value)) {
        return <span aria-label={name}>{value ? t('common.labels.on') : t('common.labels.off')}</span>;
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
            isDisabled={isDisabled}
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

type ParameterFieldProps = {
    parameter: ConfigurableParameter;
    onChange: (parameter: ConfigurableParameter) => void;
    isDisabled?: boolean;
};

const ParameterField = ({ parameter, onChange, isDisabled }: ParameterFieldProps) => {
    if (isEnumStringParameter(parameter)) {
        return <StringEnumParameterField parameter={parameter} onChange={onChange} isDisabled={isDisabled} />;
    }

    if (isEnumNumberParameter(parameter)) {
        return <NumberEnumParameterField parameter={parameter} onChange={onChange} isDisabled={isDisabled} />;
    }

    if (isNumberParameter(parameter)) {
        return (
            <NumberParameterField
                value={parameter.value}
                minValue={parameter.min_value ?? null}
                maxValue={parameter.max_value ?? null}
                onChange={(value) => onChange({ ...parameter, value })}
                type={parameter.value_type}
                isDisabled={isDisabled}
                name={parameter.name}
            />
        );
    }

    if (parameter.value_type === 'float_range') {
        return (
            <RangeParameterField
                value={parameter.value}
                maxValue={parameter.max_value}
                minValue={parameter.min_value}
                onChange={(value) => onChange({ ...parameter, value })}
                isDisabled={isDisabled}
                name={parameter.name}
            />
        );
    }

    if (parameter.value_type === 'bool') {
        return (
            <BooleanParameterField
                value={parameter.value}
                header={parameter.name}
                onChange={(value) => onChange({ ...parameter, value })}
                isDisabled={isDisabled}
            />
        );
    }

    return null;
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

const createTestId = (keys: string[] | undefined, parameterKey: string) =>
    keys === undefined ? undefined : [...keys, parameterKey].join('-');

type SingleParameterProps = Omit<ParameterProps, 'header' | 'description' | 'isGroupHeader'> & { id?: string };

const SingleParameter = ({ id, parameter, onChange, isReadOnly, isDisabled, marginStart }: SingleParameterProps) => (
    <ParametersContainer id={id} isReadOnly={isReadOnly}>
        <Parameter
            header={parameter.name}
            description={parameter.description}
            parameter={parameter}
            onChange={onChange}
            isReadOnly={isReadOnly}
            isDisabled={isDisabled}
            marginStart={marginStart}
        />
    </ParametersContainer>
);

type ParametersEnableGroupProps = Omit<SharedParametersProps, 'marginStart'> & {
    group: ParametersEnableGroupParameters;
    isReadOnly: boolean;
};

// A group whose first parameter is a boolean that enables/disables the remaining ones
const ParametersEnableGroup = ({
    group,
    onChange,
    isReadOnly,
    isDisabled,
    parentGroupKeys = [],
}: ParametersEnableGroupProps) => {
    const groupKeys = [...parentGroupKeys, group.key];
    const [enableParameter, ...configurableParameters] = group.parameters;

    return (
        <View UNSAFE_className={classes.parameterGroupContainer}>
            <ParametersContainer
                rowGap={configurableParameters.length > 0 ? 'size-100' : 'size-0'}
                isReadOnly={isReadOnly}
                id={createTestId(parentGroupKeys, group.key)}
            >
                <Parameter
                    header={group.name}
                    description={group.description}
                    parameter={enableParameter}
                    onChange={(parameter) => onChange(parameter, groupKeys)}
                    isReadOnly={isReadOnly}
                    isDisabled={isDisabled}
                    isGroupHeader
                />

                <Parameters
                    parameters={configurableParameters}
                    onChange={onChange}
                    isReadOnly={isReadOnly}
                    isDisabled={isDisabled || !enableParameter.value}
                    marginStart={'size-200'}
                    parentGroupKeys={groupKeys}
                />
            </ParametersContainer>
        </View>
    );
};

type ParametersGroupProps = SharedParametersProps & { group: ConfigurableParameterGroup };

const ParametersGroup = ({
    group,
    onChange,
    isReadOnly = false,
    isDisabled,
    marginStart,
    parentGroupKeys = [],
}: ParametersGroupProps) => {
    if (isBoolEnableParameterGroup(group)) {
        return (
            <ParametersEnableGroup
                group={group}
                onChange={onChange}
                isReadOnly={isReadOnly}
                isDisabled={isDisabled}
                parentGroupKeys={parentGroupKeys}
            />
        );
    }

    return (
        <View UNSAFE_className={classes.parameterGroupContainer}>
            <Text UNSAFE_className={classes.parameterGroupTitle}>{group.name}</Text>
            <Flex direction={'column'} gap={'size-300'}>
                <Parameters
                    parameters={group.parameters}
                    onChange={onChange}
                    isReadOnly={isReadOnly}
                    isDisabled={isDisabled}
                    marginStart={marginStart}
                    parentGroupKeys={[...parentGroupKeys, group.key]}
                />
            </Flex>
        </View>
    );
};

type ParametersProps = SharedParametersProps & { parameters: TrainingConfigurationParameter[] };

export const Parameters = ({
    parameters,
    onChange,
    isReadOnly = false,
    isDisabled = false,
    marginStart,
    parentGroupKeys,
}: ParametersProps) => (
    <>
        {parameters.map((parameter) =>
            isParameter(parameter) ? (
                <SingleParameter
                    key={parameter.key}
                    id={createTestId(parentGroupKeys, parameter.key)}
                    parameter={parameter}
                    onChange={(changedParameter) => onChange(changedParameter, parentGroupKeys)}
                    isReadOnly={isReadOnly}
                    isDisabled={isDisabled}
                    marginStart={marginStart}
                />
            ) : (
                <ParametersGroup
                    key={parameter.key}
                    group={parameter}
                    onChange={onChange}
                    isReadOnly={isReadOnly}
                    isDisabled={isDisabled}
                    marginStart={marginStart}
                    parentGroupKeys={parentGroupKeys}
                />
            )
        )}
    </>
);

Parameters.Container = ParametersContainer;
Parameter.Layout = ParameterLayout;
