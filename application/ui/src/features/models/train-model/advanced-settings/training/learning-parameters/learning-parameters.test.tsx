// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import type { NumberConfigurableParameter, TrainingConfiguration } from '@/api/types';
import { fireEvent, screen, Screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from 'test-utils/render';

import { getStep } from '../../components/number-parameter-field/utils';
import { isBoolEnableParameterGroup, isNumberParameter } from '../../utils';
import { LearningParameters } from './learning-parameters.component';
import { learningParameters } from './mocks';
import {
    getInputSizeHeightParameter,
    getInputSizeWidthParameter,
    getLearningParameters,
    isInputSizeHeightParameter,
    isInputSizeParameter,
    LearningConfigurationGroup,
} from './utils';

const getParameter = (name: string, selector: Screen | ReturnType<typeof within> = screen) => {
    return selector.queryByRole('textbox', { name: `Change ${name}` });
};

const resetParameter = (name: string, selector: Screen | ReturnType<typeof within> = screen) => {
    fireEvent.click(selector.getByRole('button', { name: `Reset ${name}` }));
};

const getEnumFieldParameter = (name: string) => {
    return screen.getByRole('button', { name: new RegExp(`Select ${name}`) });
};

const expectNumberParameter = async (parameter: NumberConfigurableParameter, groupKey: string) => {
    const step = getStep({
        type: parameter.value_type,
        maxValue: parameter.max_value ?? null,
        minValue: parameter.min_value ?? null,
    });

    let selector = document.body;
    if (parameter.key === 'patience') {
        selector = screen.getByTestId(`${groupKey}-patience`);
    }

    const wrapper = within(selector);

    const parameterInput = getParameter(parameter.name, wrapper);

    expect(parameterInput).toHaveValue(parameter.value.toString());

    // Round to avoid floating-point noise (e.g. 0.004 + 0.0001).
    const newValue = Math.round((parameter.value + step) * 1e6) / 1e6;

    await userEvent.click(parameterInput as HTMLElement);

    if (parameter.value_type === 'int') {
        await userEvent.keyboard('{ArrowUp}');
    } else {
        await userEvent.clear(parameterInput as HTMLElement);
        await userEvent.type(parameterInput as HTMLElement, newValue.toString());
        await userEvent.tab();
    }

    expect(parameterInput).toHaveValue(newValue.toString());

    resetParameter(parameter.name, wrapper);
    expect(getParameter(parameter.name, wrapper)).toHaveValue(parameter.default_value.toString());
};

const expectDisabledNumberParameter = (
    parameter: NumberConfigurableParameter,
    selector: Screen | ReturnType<typeof within> = screen
) => {
    expect(getParameter(parameter.name, selector)).toBeDisabled();
};

describe('LearningParameters', () => {
    const App = (props: { learningParameters: LearningConfigurationGroup }) => {
        const [trainingConfiguration, setTrainingConfiguration] = useState<TrainingConfiguration | undefined>({
            parameters: [props.learningParameters],
        });

        return (
            <LearningParameters
                learningParameters={
                    getLearningParameters(trainingConfiguration ?? { parameters: [] }) ?? props.learningParameters
                }
                defaultLearningParameters={props.learningParameters}
                onTrainingConfigurationChange={setTrainingConfiguration}
            />
        );
    };

    it('updates tag to "Modified" when at least one parameter is changed, otherwise is "Default"', async () => {
        render(<App learningParameters={learningParameters} />);

        const maxEpochsParameter = learningParameters.parameters[0] as NumberConfigurableParameter;

        expect(screen.getByLabelText('Learning parameters tag')).toHaveTextContent('Default');

        const maxEpochsInput = getParameter(maxEpochsParameter.name);

        expect(maxEpochsInput).toHaveValue(maxEpochsParameter.value.toString());

        await userEvent.click(maxEpochsInput as HTMLElement);
        await userEvent.keyboard('{ArrowUp}');

        expect(maxEpochsInput).toHaveValue((maxEpochsParameter.value + 1).toString());
        expect(screen.getByLabelText('Learning parameters tag')).toHaveTextContent('Modified');
    });

    it('shows dependent parameters based on "scheduler type" value', () => {
        render(<App learningParameters={learningParameters} />);

        const schedulerType = getEnumFieldParameter('Scheduler type');

        expect(schedulerType).toHaveTextContent('reduce_lr_on_plateau');

        expect(getParameter('Factor')).toBeInTheDocument();
        expect(getParameter('Patience', within(screen.getByTestId('scheduler-patience')))).toBeInTheDocument();
        expect(getParameter('Minimum learning rate')).not.toBeInTheDocument();

        fireEvent.click(schedulerType);

        fireEvent.click(screen.getByRole('option', { name: 'cosine_annealing' }));

        expect(schedulerType).toHaveTextContent('cosine_annealing');

        expect(getParameter('Factor')).not.toBeInTheDocument();
        expect(screen.queryByTestId('scheduler-patience')).not.toBeInTheDocument();
        expect(getParameter('Minimum learning rate')).toBeInTheDocument();
    });

    describe('updates parameters and resets them to default properly', () => {
        const parametersWithoutInputSizeParameters = {
            ...learningParameters,
            parameters: learningParameters.parameters.filter((parameter) => !isInputSizeParameter(parameter)),
        };

        const numberParameters = parametersWithoutInputSizeParameters.parameters.filter(isNumberParameter);
        const enableGroupParameters =
            parametersWithoutInputSizeParameters.parameters.filter(isBoolEnableParameterGroup);

        it('updates and resets number parameters', async () => {
            render(<App learningParameters={parametersWithoutInputSizeParameters} />);

            for (const parameter of numberParameters) {
                await expectNumberParameter(parameter, parametersWithoutInputSizeParameters.key);
            }
        });

        it.each(enableGroupParameters.map((p) => [p.name, p] as const))(
            'updates and resets enable group: %s',
            (_name, parameter) => {
                render(<App learningParameters={parametersWithoutInputSizeParameters} />);

                const [enableParameter, ...restParameters] = parameter.parameters;

                const groupContainer = within(screen.getByTestId(parameter.key));

                const toggleEnableParameter = groupContainer.getByRole('switch', {
                    name: `Toggle ${enableParameter.name}`,
                });

                if (!enableParameter.value) {
                    fireEvent.click(toggleEnableParameter);
                }

                expect(toggleEnableParameter).toBeChecked();

                fireEvent.click(toggleEnableParameter);

                expect(toggleEnableParameter).not.toBeChecked();

                for (const restParameter of restParameters) {
                    if (isNumberParameter(restParameter)) {
                        expectDisabledNumberParameter(restParameter, groupContainer);
                    }
                }

                fireEvent.click(screen.getByRole('button', { name: `Reset ${parameter.name}` }));
            }
        );
    });

    describe('Input size parameters', () => {
        it('updates input size parameters', () => {
            render(<App learningParameters={learningParameters} />);

            const inputSizeWidthParameter = getInputSizeWidthParameter(learningParameters.parameters);
            const inputSizeHeightParameter = getInputSizeHeightParameter(learningParameters.parameters);

            if (!inputSizeWidthParameter || !inputSizeHeightParameter) {
                throw new Error('Input size parameters not found');
            }

            const parameters = [inputSizeWidthParameter, inputSizeHeightParameter];

            parameters.forEach((parameter) => {
                const parameterSelector = getEnumFieldParameter(parameter.name);

                expect(parameterSelector).toHaveTextContent(parameter.value.toString());

                fireEvent.click(parameterSelector);

                parameter.allowed_values.forEach((value) => {
                    expect(screen.getByRole('option', { name: value.toString() })).toBeInTheDocument();
                });

                fireEvent.keyDown(screen.getByRole('listbox', { name: `Select ${parameter.name}` }), { key: 'Escape' });
            });

            fireEvent.click(getEnumFieldParameter(inputSizeWidthParameter.name));
            fireEvent.click(screen.getByRole('option', { name: '1024' }));
            expect(getEnumFieldParameter(inputSizeWidthParameter.name)).toHaveTextContent('1024');

            fireEvent.click(getEnumFieldParameter(inputSizeHeightParameter.name));
            fireEvent.click(screen.getByRole('option', { name: '256' }));
            expect(getEnumFieldParameter(inputSizeHeightParameter.name)).toHaveTextContent('256');
        });

        it('does not display input size parameters if there is only one of them', () => {
            const newLearningParameters = {
                ...learningParameters,
                parameters: learningParameters.parameters.filter((parameter) => !isInputSizeHeightParameter(parameter)),
            };

            render(<App learningParameters={newLearningParameters} />);

            expect(
                screen.queryByRole('button', { name: new RegExp(`Select Input size width`) })
            ).not.toBeInTheDocument();
            expect(
                screen.queryByRole('button', { name: new RegExp(`Select Input size height`) })
            ).not.toBeInTheDocument();
        });

        it('resets both input size parameters to default values', () => {
            const inputSizeWidthParameter = getInputSizeWidthParameter(learningParameters.parameters);
            const inputSizeHeightParameter = getInputSizeHeightParameter(learningParameters.parameters);

            if (!inputSizeWidthParameter || !inputSizeHeightParameter) {
                throw new Error('Input size parameters not found');
            }

            render(<App learningParameters={learningParameters} />);

            fireEvent.click(getEnumFieldParameter(inputSizeWidthParameter.name));
            fireEvent.click(screen.getByRole('option', { name: '1024' }));

            expect(getEnumFieldParameter(inputSizeWidthParameter.name)).toHaveTextContent('1024');

            fireEvent.click(getEnumFieldParameter(inputSizeHeightParameter.name));
            fireEvent.click(screen.getByRole('option', { name: '1024' }));

            expect(getEnumFieldParameter(inputSizeHeightParameter.name)).toHaveTextContent('1024');

            fireEvent.click(screen.getByRole('button', { name: 'Reset Input size' }));

            expect(getEnumFieldParameter(inputSizeWidthParameter.name)).toHaveTextContent(
                inputSizeWidthParameter.default_value.toString()
            );
            expect(getEnumFieldParameter(inputSizeHeightParameter.name)).toHaveTextContent(
                inputSizeHeightParameter.default_value.toString()
            );
        });
    });
});
