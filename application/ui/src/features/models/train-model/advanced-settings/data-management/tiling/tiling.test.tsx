// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import type { TrainingConfiguration } from '@/api/types';
import { createI18nInstance } from '@/i18n';
import { fireEvent, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {
    getMockedConfigurationParameter,
    getMockedConfigurationParameterGroup,
} from 'mocks/mock-training-configuration';
import { render } from 'test-utils/render';

import { Tiling } from './tiling.component';
import {
    getTilingAutomaticDescription,
    getTilingMode,
    getTilingOffDescription,
    getTilingParameters,
    TILING_MODES,
    TilingConfigurableParameterGroup,
    TilingMode,
} from './utils';

describe('getTilingMode', () => {
    it('tiling is off when enable tiling parameter is absent', () => {
        expect(getTilingMode([])).toBe(TILING_MODES.OFF);
    });

    it('tiling is off when enable tiling parameter is false', () => {
        expect(
            getTilingMode([
                getMockedConfigurationParameter({
                    key: 'enable',
                    value_type: 'bool',
                    name: 'Enable tiling',
                    value: false,
                    description: 'Whether to apply tiling to the image',
                    default_value: false,
                }),
            ])
        ).toBe(TILING_MODES.OFF);
    });

    it('tiling is automatic when adaptive tiling parameter is true', () => {
        expect(
            getTilingMode([
                getMockedConfigurationParameter({
                    key: 'enable',
                    value_type: 'bool',
                    name: 'Enable tiling',
                    value: true,
                    description: 'Whether to apply tiling to the image',
                    default_value: false,
                }),
                getMockedConfigurationParameter({
                    key: 'enable_adaptive_tiling',
                    value_type: 'bool',
                    name: 'Adaptive tiling',
                    value: true,
                    description: 'Whether to use adaptive tiling based on image content',
                    default_value: false,
                }),
            ])
        ).toBe(TILING_MODES.AUTOMATIC);
    });

    it('tiling is custom when adaptive tiling parameter is false and enable tiling parameter is true', () => {
        expect(
            getTilingMode([
                getMockedConfigurationParameter({
                    key: 'enable',
                    value_type: 'bool',
                    name: 'Enable tiling',
                    value: true,
                    description: 'Whether to apply tiling to the image',
                    default_value: false,
                }),
                getMockedConfigurationParameter({
                    key: 'enable_adaptive_tiling',
                    value_type: 'bool',
                    name: 'Adaptive tiling',
                    value: false,
                    description: 'Whether to use adaptive tiling based on image content',
                    default_value: false,
                }),
            ])
        ).toBe(TILING_MODES.CUSTOM);
    });
});

const getTilingModeButton = (tilingMode: TilingMode) => {
    return screen.getByRole('button', { name: new RegExp(tilingMode, 'i') });
};

describe('Tiling', () => {
    const { t } = createI18nInstance({ lng: 'en' });

    const customParameters = [
        getMockedConfigurationParameter({
            key: 'tile_size',
            value_type: 'int',
            name: 'Tile size',
            value: 256,
            description: 'Size of each tile in pixels',
            default_value: 128,
            max_value: null,
            min_value: 0,
        }),
        getMockedConfigurationParameter({
            key: 'tile_overlap',
            value_type: 'int',
            name: 'Tile overlap',
            value: 64,
            description: 'Overlap between adjacent tiles in pixels',
            default_value: 64,
            max_value: null,
            min_value: 0,
        }),
    ];

    const tilingParameters = getMockedConfigurationParameterGroup({
        key: 'tiling',
        name: 'Tiling',
        parameters: [
            getMockedConfigurationParameter({
                key: 'enable',
                value_type: 'bool',
                name: 'Enable tiling',
                value: true,
                description: 'Whether to apply tiling to the image',
                default_value: false,
            }),
            getMockedConfigurationParameter({
                key: 'enable_adaptive_tiling',
                value_type: 'bool',
                name: 'Adaptive tiling',
                value: false,
                description: 'Whether to use adaptive tiling based on image content',
                default_value: false,
            }),
            ...customParameters,
        ],
    }) as TilingConfigurableParameterGroup;

    const App = (props: { tilingParameters: TilingConfigurableParameterGroup }) => {
        const [trainingConfiguration, setTrainingConfiguration] = useState<TrainingConfiguration | undefined>({
            parameters: [
                getMockedConfigurationParameterGroup({
                    key: 'dataset_preparation',
                    parameters: [
                        getMockedConfigurationParameterGroup({
                            key: 'augmentation',
                            parameters: [props.tilingParameters],
                        }),
                    ],
                }),
            ],
        });

        return (
            <Tiling
                tilingParameters={
                    getTilingParameters(trainingConfiguration ?? { parameters: [] }) ?? props.tilingParameters
                }
                onTrainingConfigurationChange={setTrainingConfiguration}
            />
        );
    };

    it('tiling off and automatic displays only description and no parameters', () => {
        render(<App tilingParameters={tilingParameters} />);

        fireEvent.click(getTilingModeButton(TILING_MODES.OFF));

        expect(getTilingModeButton(TILING_MODES.OFF)).toHaveAttribute('aria-pressed', 'true');
        expect(getTilingModeButton(TILING_MODES.AUTOMATIC)).toHaveAttribute('aria-pressed', 'false');
        expect(getTilingModeButton(TILING_MODES.CUSTOM)).toHaveAttribute('aria-pressed', 'false');
        expect(screen.getByText(getTilingOffDescription(t))).toBeInTheDocument();

        fireEvent.click(getTilingModeButton(TILING_MODES.AUTOMATIC));

        expect(getTilingModeButton(TILING_MODES.AUTOMATIC)).toHaveAttribute('aria-pressed', 'true');
        expect(getTilingModeButton(TILING_MODES.OFF)).toHaveAttribute('aria-pressed', 'false');
        expect(getTilingModeButton(TILING_MODES.CUSTOM)).toHaveAttribute('aria-pressed', 'false');
        expect(screen.getByText(getTilingAutomaticDescription(t))).toBeInTheDocument();
    });

    it('tiling tag updates properly when tiling mode changes', () => {
        render(<App tilingParameters={tilingParameters} />);

        fireEvent.click(getTilingModeButton(TILING_MODES.CUSTOM));

        expect(screen.getByLabelText('Tiling tag')).toHaveTextContent(TILING_MODES.CUSTOM);

        fireEvent.click(getTilingModeButton(TILING_MODES.OFF));

        expect(screen.getByLabelText('Tiling tag')).toHaveTextContent(TILING_MODES.OFF);

        fireEvent.click(getTilingModeButton(TILING_MODES.AUTOMATIC));

        expect(screen.getByLabelText('Tiling tag')).toHaveTextContent(TILING_MODES.AUTOMATIC);
    });

    it('custom tiling parameters updates and resets properly', async () => {
        render(<App tilingParameters={tilingParameters} />);

        fireEvent.click(getTilingModeButton(TILING_MODES.CUSTOM));

        expect(getTilingModeButton(TILING_MODES.CUSTOM)).toHaveAttribute('aria-pressed', 'true');
        expect(getTilingModeButton(TILING_MODES.AUTOMATIC)).toHaveAttribute('aria-pressed', 'false');
        expect(getTilingModeButton(TILING_MODES.OFF)).toHaveAttribute('aria-pressed', 'false');

        for (const parameter of customParameters) {
            const parameterInput = screen.getByRole('textbox', { name: `Change ${parameter.name}` });

            expect(parameterInput).toHaveValue(parameter.value.toString());

            await userEvent.clear(parameterInput);
            await userEvent.type(parameterInput, (Number(parameter.value) + 1).toString());
            expect(parameterInput).toHaveValue((Number(parameter.value) + 1).toString());

            await userEvent.click(screen.getByRole('button', { name: `Reset ${parameter.name}` }));
            expect(parameterInput).toHaveValue(parameter.default_value.toString());
        }
    });
});
