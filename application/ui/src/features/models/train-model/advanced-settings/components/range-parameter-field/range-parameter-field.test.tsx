// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { fireEvent, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from 'test-utils/render';

import { RangeParameterField } from './range-parameter-field.component';

describe('RangeParameterField', () => {
    const name = 'Scaling ratio range';
    const onChange = rs.fn();

    beforeEach(() => {
        rs.clearAllMocks();
    });

    const renderApp = ({
        value = [0.5, 1.5],
        isDisabled = false,
    }: {
        value?: [number, number];
        isDisabled?: boolean;
    }) => {
        return render(
            <RangeParameterField
                value={value}
                onChange={onChange}
                name={name}
                isDisabled={isDisabled}
                step={0.001}
                minValue={0}
                maxValue={2}
            />
        );
    };

    it('renders two NumberFields and a RangeSlider', () => {
        renderApp({});

        expect(screen.getByLabelText(`Change ${name} start range value`)).toBeInTheDocument();
        expect(screen.getByLabelText(`Change ${name} end range value`)).toBeInTheDocument();
        expect(screen.getByLabelText(`Change ${name} value`)).toBeInTheDocument();
    });

    it('calls onChange when start value changes', async () => {
        renderApp({});

        const startField = screen.getByLabelText(`Change ${name} start range value`);
        await userEvent.clear(startField);
        await userEvent.type(startField, '1');
        // Simulate blur to trigger onChange
        startField.blur();

        expect(onChange).toHaveBeenCalledWith([1, 1.5]);
    });

    it('calls onChange when end value changes', async () => {
        renderApp({});

        const endField = screen.getByLabelText(`Change ${name} end range value`);
        await userEvent.clear(endField);
        await userEvent.type(endField, '1');
        endField.blur();

        expect(onChange).toHaveBeenCalledWith([0.5, 1]);
    });

    it('allows start and end to be equal', async () => {
        renderApp({});

        const startField = screen.getByLabelText(`Change ${name} start range value`);
        await userEvent.clear(startField);
        await userEvent.type(startField, '1');
        startField.blur();

        const endField = screen.getByLabelText(`Change ${name} end range value`);
        await userEvent.clear(endField);
        await userEvent.type(endField, '1');
        endField.blur();

        expect(onChange).toHaveBeenCalledWith([1, 1]);
    });

    it('disables fields when isDisabled is true', () => {
        renderApp({ isDisabled: true });

        expect(screen.getByLabelText(`Change ${name} start range value`)).toBeDisabled();
        expect(screen.getByLabelText(`Change ${name} end range value`)).toBeDisabled();
    });

    it('does not allow start value to exceed end value', async () => {
        renderApp({ value: [0.5, 1.5] });

        const startField = screen.getByLabelText(`Change ${name} start range value`);
        await userEvent.clear(startField);
        await userEvent.type(startField, '2');
        startField.blur();

        expect(onChange).toHaveBeenCalledWith([1.5, 1.5]);
        expect(onChange).not.toHaveBeenCalledWith([2, 1.5]);

        await waitFor(() => {
            expect(screen.getByLabelText(`Change ${name} start range value`)).toHaveValue('1.5');
            expect(screen.getByLabelText(`Change ${name} end range value`)).toHaveValue('1.5');
        });
    });

    it('does not allow end value to go below start value', async () => {
        renderApp({ value: [0.5, 1.5] });

        const endField = screen.getByLabelText(`Change ${name} end range value`);
        await userEvent.clear(endField);
        await userEvent.type(endField, '0');
        endField.blur();

        expect(onChange).toHaveBeenCalledWith([0.5, 0.5]);
        expect(onChange).not.toHaveBeenCalledWith([0.5, 0]);

        await waitFor(() => {
            expect(screen.getByLabelText(`Change ${name} start range value`)).toHaveValue('0.5');
            expect(screen.getByLabelText(`Change ${name} end range value`)).toHaveValue('0.5');
        });
    });

    it('syncs internal state when value prop changes externally', () => {
        const ControlledWrapper = () => {
            const [value, setValue] = useState<[number, number]>([0.5, 1.5]);

            return (
                <>
                    <button onClick={() => setValue([0.2, 1.8])}>Update value</button>
                    <RangeParameterField
                        value={value}
                        onChange={onChange}
                        name={name}
                        step={0.001}
                        minValue={0}
                        maxValue={2}
                    />
                </>
            );
        };

        render(<ControlledWrapper />);

        expect(screen.getByLabelText(`Change ${name} start range value`)).toHaveValue('0.5');
        expect(screen.getByLabelText(`Change ${name} end range value`)).toHaveValue('1.5');

        fireEvent.click(screen.getByRole('button', { name: 'Update value' }));

        expect(screen.getByLabelText(`Change ${name} start range value`)).toHaveValue('0.2');
        expect(screen.getByLabelText(`Change ${name} end range value`)).toHaveValue('1.8');
    });
});
