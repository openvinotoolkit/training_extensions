// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { ConfigurableParameter, NumberEnumConfigurableParameter } from '@/api/types';
import { useTranslation } from '@/i18n';
import { Flex } from '@geti-ui/ui';

import { NumberEnumParameterField, Parameter, Parameters } from '../../components/parameters.component';
import { getInputSizeHeightParameter, getInputSizeWidthParameter } from './utils';

type InputSizeParameterProps = {
    inputSizeParameter: NumberEnumConfigurableParameter;
    onChange: (parameter: NumberEnumConfigurableParameter) => void;
    isReadOnly?: boolean;
};

const InputSizeParameter = ({ inputSizeParameter, onChange, isReadOnly }: InputSizeParameterProps) => {
    if (isReadOnly || inputSizeParameter.allowed_values.length <= 1) {
        return <span aria-label={inputSizeParameter.name}>{inputSizeParameter.value}</span>;
    }

    return <NumberEnumParameterField parameter={inputSizeParameter} onChange={onChange} />;
};

type InputSizeParametersProps = {
    inputSizeParameters: ConfigurableParameter[];
    isReadOnly?: boolean;
    onInputSizeParameterChange: (parameter: NumberEnumConfigurableParameter[]) => void;
};

export const InputSizeParameters = ({
    inputSizeParameters,
    onInputSizeParameterChange,
    isReadOnly = false,
}: InputSizeParametersProps) => {
    const { t } = useTranslation();
    const inputSizeWidthParameter = getInputSizeWidthParameter(inputSizeParameters);
    const inputSizeHeightParameter = getInputSizeHeightParameter(inputSizeParameters);

    if (inputSizeWidthParameter === undefined || inputSizeHeightParameter === undefined) return null;

    const description = `${inputSizeWidthParameter.description}\n${inputSizeHeightParameter.description}`;

    const handleReset = () => {
        onInputSizeParameterChange([
            {
                ...inputSizeWidthParameter,
                value: inputSizeWidthParameter.default_value,
            },
            {
                ...inputSizeHeightParameter,
                value: inputSizeHeightParameter.default_value,
            },
        ]);
    };

    const handleInputSizeParameterChange = (parameter: NumberEnumConfigurableParameter) => {
        onInputSizeParameterChange([parameter]);
    };

    return (
        <Parameters.Container>
            <Parameter.Layout
                header={t('models.training.learning.inputSize')}
                ariaLabel={'Input size'}
                description={description}
                onReset={handleReset}
            >
                <Flex alignItems={'center'} gap={'size-50'}>
                    <InputSizeParameter
                        inputSizeParameter={inputSizeWidthParameter}
                        onChange={handleInputSizeParameterChange}
                        isReadOnly={isReadOnly}
                    />
                    x
                    <InputSizeParameter
                        inputSizeParameter={inputSizeHeightParameter}
                        onChange={handleInputSizeParameterChange}
                        isReadOnly={isReadOnly}
                    />
                </Flex>
            </Parameter.Layout>
        </Parameters.Container>
    );
};
