// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from 'test-utils/render';

import { ConfidenceThreshold } from './confidence-threshold.component';

const getInput = () => screen.getByRole('textbox', { name: 'Change Confidence threshold' });
const getResetButton = () => screen.getByRole('button', { name: 'Reset confidence threshold' });

describe('ConfidenceThreshold', () => {
    it('renders the provided value', () => {
        render(<ConfidenceThreshold value={0.65} defaultValue={0.65} onChange={vi.fn()} />);

        expect(getInput()).toHaveValue('0.65');
    });

    it('calls onChange when the value is committed in the numeric input', async () => {
        const onChange = vi.fn();
        render(<ConfidenceThreshold value={0.65} defaultValue={0.65} onChange={onChange} />);

        await userEvent.clear(getInput());
        await userEvent.type(getInput(), '0.8');
        await userEvent.tab();

        expect(onChange).toHaveBeenCalledWith(0.8);
    });

    it('does not call onChange while the user is still typing', async () => {
        const onChange = vi.fn();
        render(<ConfidenceThreshold value={0.65} defaultValue={0.65} onChange={onChange} />);

        await userEvent.clear(getInput());
        await userEvent.type(getInput(), '0.8');

        expect(onChange).not.toHaveBeenCalled();
    });

    it('commits the numeric input when Enter is pressed', async () => {
        const onChange = vi.fn();
        render(<ConfidenceThreshold value={0.65} defaultValue={0.65} onChange={onChange} />);

        await userEvent.clear(getInput());
        await userEvent.type(getInput(), '0.8{Enter}');

        expect(onChange).toHaveBeenCalledExactlyOnceWith(0.8);
    });

    it('restores the default value when reset is pressed', async () => {
        const onChange = vi.fn();
        render(<ConfidenceThreshold value={0.8} defaultValue={0.65} onChange={onChange} />);

        await userEvent.click(getResetButton());

        expect(onChange).toHaveBeenCalledWith(0.65);
    });

    it('disables reset when the value already equals the default', () => {
        render(<ConfidenceThreshold value={0.65} defaultValue={0.65} onChange={vi.fn()} />);

        expect(getResetButton()).toBeDisabled();
    });
});
