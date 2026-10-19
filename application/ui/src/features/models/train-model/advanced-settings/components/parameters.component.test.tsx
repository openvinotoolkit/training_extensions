// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import {
    getMockedConfigurationParameter,
    getMockedConfigurationParameterGroup,
} from 'mocks/mock-training-configuration';
import { render } from 'test-utils/render';

import { Parameters } from './parameters.component';

const numberEnumParameter = getMockedConfigurationParameter({
    key: 'input_size_width',
    name: 'Input size width',
    description: 'Width',
    depends_on: null,
    value_type: 'int',
    value: 992,
    default_value: 992,
    min_value: 0,
    max_value: null,
    allowed_values: [128, 256, 384, 512, 640, 800, 992, 1024],
});

const getEnableGroup = (isEnabled: boolean) =>
    getMockedConfigurationParameterGroup({
        key: 'early_stopping',
        name: 'Early stopping',
        description: 'Early stopping',
        parameters: [
            getMockedConfigurationParameter({
                value_type: 'bool',
                key: 'enable',
                name: 'Enable',
                description: 'Enable',
                depends_on: null,
                value: isEnabled,
                default_value: isEnabled,
            }),
            getMockedConfigurationParameter({
                value_type: 'int',
                key: 'patience',
                name: 'Patience',
                description: 'Patience',
                depends_on: null,
                value: 10,
                default_value: 10,
                min_value: 1,
                max_value: null,
                allowed_values: null,
            }),
        ],
    });

const getPatienceField = () => screen.getByRole('textbox', { name: 'Change Patience' });

describe('Parameters', () => {
    it('disables the picker of a number enum parameter with many allowed values', () => {
        render(<Parameters parameters={[numberEnumParameter]} onChange={rs.fn()} isDisabled />);

        expect(screen.getByRole('button', { name: /Select Input size width/ })).toBeDisabled();
    });

    it('disables the parameters of an enable group when the group is toggled off', () => {
        render(<Parameters parameters={[getEnableGroup(false)]} onChange={rs.fn()} />);

        expect(getPatienceField()).toBeDisabled();
    });

    it('keeps the parameters of an enabled group disabled when its parent is disabled', () => {
        render(<Parameters parameters={[getEnableGroup(true)]} onChange={rs.fn()} isDisabled />);

        expect(getPatienceField()).toBeDisabled();
    });

    it('enables the parameters of an enabled group when its parent is enabled', () => {
        render(<Parameters parameters={[getEnableGroup(true)]} onChange={rs.fn()} />);

        expect(getPatienceField()).toBeEnabled();
    });
});
